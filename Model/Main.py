'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utility.helper import *
from utility.batch_test import *
from time import time

from BPRMF import BPRMF
from CKE import CKE
from CFKG import CFKG
from NFM import NFM
from KGAT import KGAT
from tqdm import tqdm
from collections import defaultdict
import heapq


import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class PathExplainer:
    """高效的路径解释器 - 寻找真实的多跳路径"""
    def __init__(self, KG):
        self.KG = KG
        self.path_cache = {}
        
    def find_multi_hop_paths(self, start, target, max_hops=3, beam_width=20):
        """寻找多跳路径，返回真实的KG路径"""
        cache_key = f"{start}_{target}_{max_hops}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        paths = []
        if start == target:
            self.path_cache[cache_key] = paths
            return paths
            
        # 使用Beam Search而不是BFS
        # beam: (node, path, score, visited)
        beam = [(start, [], 1.0, {start})]
        
        for hop in range(max_hops):
            next_beam = []
            
            for node, path, score, visited in beam:
                if node not in self.KG:
                    continue
                
                # 获取邻居并按attention排序
                neighbors = self.KG.get(node, [])
                if not neighbors:
                    continue
                    
                neighbors.sort(key=lambda x: -x[2])
                
                # 取top-k邻居
                for neighbor, relation, weight in neighbors[:10]:
                    if neighbor in visited:
                        continue
                    
                    # 创建新路径
                    new_path = path + [(node, relation, neighbor, weight)]
                    new_score = score * weight
                    new_visited = visited.copy()
                    new_visited.add(neighbor)
                    
                    # 检查是否到达目标
                    if neighbor == target:
                        paths.append({
                            "score": new_score,
                            "path": new_path
                        })
                        # 如果找到足够路径，可以提前返回
                        if len(paths) >= 5:
                            break
                    else:
                        next_beam.append((neighbor, new_path, new_score, new_visited))
            
            # 对next_beam按score排序并保留beam_width个
            next_beam.sort(key=lambda x: -x[2])
            beam = next_beam[:beam_width]
            
            if not beam:
                break
        
        # 按分数排序
        paths.sort(key=lambda x: -x["score"])
        self.path_cache[cache_key] = paths
        return paths
    
    def find_paths_via_intermediate_entities(self, start_items, target_item, max_hops=3):
        """通过中间实体寻找路径"""
        all_paths = []
        
        for start in start_items:
            # 寻找所有可能路径
            paths = self.find_multi_hop_paths(start, target_item, max_hops)
            
            for p in paths:
                if p["path"]:  # 确保路径不为空
                    p["start_item"] = start
                    all_paths.append(p)
        
        # 如果没有找到路径，尝试寻找通过共同邻居的路径
        if not all_paths:
            all_paths = self.find_paths_via_common_entities(start_items, target_item)
        
        # 去重并排序
        unique_paths = []
        seen = set()
        
        for p in all_paths:
            # 创建路径的唯一标识
            path_key = tuple((h, r, t) for h, r, t, _ in p["path"])
            if path_key not in seen:
                seen.add(path_key)
                unique_paths.append(p)
        
        unique_paths.sort(key=lambda x: -x["score"])
        return unique_paths[:1]  # 返回最多5条路径
    
    def find_paths_via_common_entities(self, start_items, target_item):
        """通过共同实体寻找路径（替代方法）"""
        paths = []
        
        if target_item not in self.KG:
            return paths
            
        # 获取目标物品的邻居
        target_neighbors = {t for t, _, _ in self.KG.get(target_item, [])}
        
        for start in start_items:
            if start not in self.KG:
                continue
                
            # 寻找start->X->target的路径
            for neighbor1, rel1, w1 in self.KG.get(start, []):
                if neighbor1 not in self.KG:
                    continue
                    
                # 从neighbor1寻找连接到target的路径
                for neighbor2, rel2, w2 in self.KG.get(neighbor1, []):
                    if neighbor2 == target_item:
                        # 找到2跳路径
                        paths.append({
                            "score": w1 * w2,
                            "path": [
                                (start, rel1, neighbor1, w1),
                                (neighbor1, rel2, target_item, w2)
                            ],
                            "start_item": start
                        })
                    elif neighbor2 in self.KG:
                        # 寻找3跳路径
                        for neighbor3, rel3, w3 in self.KG.get(neighbor2, []):
                            if neighbor3 == target_item:
                                paths.append({
                                    "score": w1 * w2 * w3,
                                    "path": [
                                        (start, rel1, neighbor1, w1),
                                        (neighbor1, rel2, neighbor2, w2),
                                        (neighbor2, rel3, target_item, w3)
                                    ],
                                    "start_item": start
                                })
        
        return paths
    


def get_topk_items_for_user(sess, model, user_id, k=10):
    """
    返回该 user 的 Top-K item id
    """
    # 构造全部 item 的索引
    all_items = np.arange(model.n_items)

    # 构造 feed_dict
    feed_dict = {
        model.users: np.array([user_id] * model.n_items),
        model.pos_items: all_items,
        model.neg_items: all_items,
        model.node_dropout: np.zeros(model.n_layers + 1),
        model.mess_dropout: np.zeros(model.n_layers + 1),
    }

    # 获取评分
    scores = sess.run(model.batch_predictions, feed_dict=feed_dict)
    scores = scores.flatten()

    # 取 top-k
    topk_items = heapq.nlargest(k, range(len(scores)), scores.__getitem__)
    return topk_items


def export_detailed_paths(sess, model, data_generator, KG, save_path):
    print("Exporting KG paths...")
    explainer = PathExplainer(KG)

    with open(save_path, "w", encoding="utf-8") as f:
        test_users = list(data_generator.test_user_dict.keys())[:50]
        print("Number of test users:", len(test_users))

        for user in tqdm(test_users, desc="Users"):
            f.write(f"User {user}\n")

            hist_items = data_generator.train_user_dict.get(user, [])
            top_items = get_topk_items_for_user(sess, model, user, k=10)

            for rank, item in enumerate(top_items, 1):
                f.write(f"  Top-{rank} Item {item}\n")

                if not hist_items:
                    f.write("    No path found (no history)\n")
                    continue

                paths = explainer.find_paths_via_intermediate_entities(
                    hist_items, item, max_hops=3
                )

                if not paths:
                    f.write("    No path found\n")
                    continue

                # ⭐ 只取第一条（已经是 score 最大）
                p = paths[0]

                f.write(f"    Path (score={p['score']:.6f})\n")

                # user → 起始历史 item
                first_h = p["path"][0][0]
                f.write(f"      (User {user}) --interact--> ({first_h})\n")

                for h, r, t, w in p["path"]:
                    f.write(f"      ({h}, {r}, {t}), attention={w:.6f}\n")

            f.write("\n")


def export_paths_for_users(sess, model, data_generator, KG, save_path, users_to_export,
                           top_k_items=10, max_paths_per_user=5):
    print(f"Exporting paths for {len(users_to_export)} users to {save_path} ...")
    explainer = PathExplainer(KG)
    with open(save_path, "w", encoding="utf-8") as f:
        for user in tqdm(users_to_export, desc="Users"):
            f.write(f"User {user}\n")
            hist_items = data_generator.train_user_dict.get(user, [])
            top_items = get_topk_items_for_user(sess, model, user, k=top_k_items)
            paths_count = 0
            for item in top_items:
                if not hist_items:
                    continue
                paths = explainer.find_paths_via_intermediate_entities(hist_items, item, max_hops=3)
                for p in paths:
                    if paths_count >= max_paths_per_user:
                        break
                    f.write(f"  Item {item}, Path (score={p['score']:.6f})\n")
                    first_h = p["path"][0][0]
                    f.write(f"    (User {user}) --interact--> ({first_h})\n")
                    for h, r, t, w in p["path"]:
                        f.write(f"    ({h}, {r}, {t}), attention={w:.6f}\n")
                    paths_count += 1
                if paths_count >= max_paths_per_user:
                    break
            f.write("\n")



def load_pretrained_data(args):
    pre_model = 'mf'
    if args.pretrain == -2:
        pre_model = 'kgat'
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, pre_model)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained bprmf model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    # get argument settings.
    tf.set_random_seed(2019)
    np.random.seed(2019)
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities

    if args.model_type in ['kgat', 'cfkg']:
        "Load the laplacian matrix."
        config['A_in'] = sum(data_generator.lap_list)

        "Load the KG triplets."
        config['all_h_list'] = data_generator.all_h_list
        config['all_r_list'] = data_generator.all_r_list
        config['all_t_list'] = data_generator.all_t_list
        config['all_v_list'] = data_generator.all_v_list

    t0 = time()

    """
    *********************************************************
    Use the pretrained data to initialize the embeddings.
    """
    if args.pretrain in [-1, -2]:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None

    """
    *********************************************************
    Select one of the models.
    """
    if args.model_type == 'bprmf':
        model = BPRMF(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type == 'cke':
        model = CKE(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type in ['cfkg']:
        model = CFKG(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type in ['nfm', 'fm']:
        model = NFM(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type in ['kgat']:
        model = KGAT(data_config=config, pretrain_data=pretrain_data, args=args)

    saver = tf.train.Saver()

    """
    *********************************************************
    Save the model parameters.
    """
    if args.save_flag == 1:
        if args.model_type in ['bprmf', 'cke', 'fm', 'cfkg']:
            weights_save_path = '%sweights/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type,
                                                             str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        elif args.model_type in ['ncf', 'nfm', 'kgat']:
            layer = '-'.join([str(l) for l in eval(args.layer_size)])
            weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (
                args.weights_path, args.dataset, model.model_type, layer, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the model parameters to fine tune.
    """
    if args.pretrain == 1:
        if args.model_type in ['bprmf', 'cke', 'fm', 'cfkg']:
            pretrain_path = '%sweights/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, str(args.lr),
                                                             '-'.join([str(r) for r in eval(args.regs)]))

        elif args.model_type in ['ncf', 'nfm', 'kgat']:
            layer = '-'.join([str(l) for l in eval(args.layer_size)])
            pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (
                args.weights_path, args.dataset, model.model_type, layer, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from the model to fine tune.
            if args.report != 1:
                users_to_test = list(data_generator.test_user_dict.keys())

                ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f], auc=[%.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
                print(pretrain_ret)

                # *********************************************************
                # save the pretrained model parameters of mf (i.e., only user & item embeddings) for pretraining other models.
                if args.save_flag == -1:
                    user_embed, item_embed = sess.run(
                        [model.weights['user_embedding'], model.weights['item_embedding']],
                        feed_dict={})
                    # temp_save_path = '%spretrain/%s/%s/%s_%s.npz' % (args.proj_path, args.dataset, args.model_type, str(args.lr),
                    #                                                  '-'.join([str(r) for r in eval(args.regs)]))
                    temp_save_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, model.model_type)
                    ensureDir(temp_save_path)
                    np.savez(temp_save_path, user_embed=user_embed, item_embed=item_embed)
                    print('save the weights of fm in path: ', temp_save_path)
                    exit()

                # *********************************************************
                # save the pretrained model parameters of kgat (i.e., user & item & kg embeddings) for pretraining other models.
                if args.save_flag == -2:
                    user_embed, entity_embed, relation_embed = sess.run(
                        [model.weights['user_embed'], model.weights['entity_embed'], model.weights['relation_embed']],
                        feed_dict={})

                    temp_save_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, args.model_type)
                    ensureDir(temp_save_path)
                    np.savez(temp_save_path, user_embed=user_embed, entity_embed=entity_embed, relation_embed=relation_embed)
                    print('save the weights of kgat in path: ', temp_save_path)
                    exit()

        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')
    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the final performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()

        users_to_test_list.append(list(data_generator.test_user_dict.keys()))
        split_state.append('all')

        save_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(save_path)
        f = open(save_path, 'w')
        f.write('embed_size=%d, lr=%.4f, regs=%s, loss_type=%s, \n' % (args.embed_size, args.lr, args.regs,
                                                                       args.loss_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in tqdm(range(args.epoch), desc="Training Epochs"):
        t1 = time()
        loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 1: to train the recommender.
        """
        for idx in tqdm(range(n_batch), desc=f"Epoch {epoch+1} Training Phase 1"):
            btime= time()

            batch_data = data_generator.generate_train_batch()
            feed_dict = data_generator.generate_train_feed_dict(model, batch_data)

            _, batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train(sess, feed_dict=feed_dict)

            loss += batch_loss
            base_loss += batch_base_loss
            kge_loss += batch_kge_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss@phase1 is nan.')
            sys.exit()

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 2: to train the KGE method & update the attentive Laplacian matrix.
        """
        if args.model_type in ['kgat']:

            n_A_batch = len(data_generator.all_h_list) // args.batch_size_kg + 1

            if args.use_kge is True:
                # using KGE method (knowledge graph embedding).
                for idx in tqdm(range(n_A_batch), desc=f"Epoch {epoch+1} Training Phase 2 (KGE)"):
                    btime = time()

                    A_batch_data = data_generator.generate_train_A_batch()
                    feed_dict = data_generator.generate_train_A_feed_dict(model, A_batch_data)

                    _, batch_loss, batch_kge_loss, batch_reg_loss = model.train_A(sess, feed_dict=feed_dict)

                    loss += batch_loss
                    kge_loss += batch_kge_loss
                    reg_loss += batch_reg_loss

            if args.use_att is True:
                # updating attentive laplacian matrix.
                model.update_attentive_A(sess)

        if np.isnan(loss) == True:
            print('ERROR: loss@phase2 is nan.')
            sys.exit()

        # show_step = 10
        # if (epoch + 1) % show_step != 0:
        #     if args.verbose > 0 and epoch % args.verbose == 0:
        #         perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
        #             epoch, time() - t1, loss, base_loss, kge_loss, reg_loss)
        #         print(perf_str)
        #     continue

        """
        *********************************************************
        Test.
        """
        t2 = time()
        users_to_test = list(data_generator.test_user_dict.keys())

        ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)

        """
        *********************************************************
        Performance logging.
        """
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        # ===== 显式转成 Python float，避免 NumPy 1.25+ warning =====
        recall_0 = float(ret['recall'][0])
        recall_1 = float(ret['recall'][-1])
        precision_0 = float(ret['precision'][0])
        precision_1 = float(ret['precision'][-1])
        hit_0 = float(ret['hit_ratio'][0])
        hit_1 = float(ret['hit_ratio'][-1])
        ndcg_0 = float(ret['ndcg'][0])
        ndcg_1 = float(ret['ndcg'][-1])

        if args.verbose > 0:
            perf_str = (
                'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], '
                'recall=[%.5f, %.5f], precision=[%.5f, %.5f], '
                'hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]'
                % (
                    epoch, t2 - t1, t3 - t2,
                    loss, base_loss, kge_loss, reg_loss,
                    recall_0, recall_1,
                    precision_0, precision_1,
                    hit_0, hit_1,
                    ndcg_0, ndcg_1
                )
            )
            print(perf_str)

        # ===== early stopping 也显式使用 float =====
        cur_best_pre_0, stopping_step, should_stop = early_stopping(
            recall_0,
            cur_best_pre_0,
            stopping_step,
            expected_order='acc',
            flag_step=10
        )

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if recall_0 == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

        # ===== 训练结束后统一转 NumPy array =====
        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)
        hit = np.array(hit_loger)
        
    """
    *********************************************************
    Export Top-10 recommendation paths with attention scores
    with dynamic max_hops expansion
    *********************************************************
    """
    
    print("Exporting top-10 recommendation paths with attention scores...")
    
    # 1. 更新注意力矩阵并获取 attention score
    att_scores = model.update_attentive_A(sess)
    
    # 2. 构建带 attention 的 KG（添加反向边以便路径搜索）
    def build_kg_with_attention(data_generator, att_scores):
        """
        构建带反向边的KG以便更好地寻找路径
        """
        KG = defaultdict(list)
        idx = 0
        
        print(f"Building KG from laplacian matrices...")
        total_edges = 0
        
        for lap, r_id in zip(data_generator.lap_list, data_generator.adj_r_list):
            coo = lap.tocoo()
            edges_in_matrix = len(coo.data)
            total_edges += edges_in_matrix
            
            for i in range(len(coo.data)):
                h = coo.row[i]
                t = coo.col[i]
                if idx < len(att_scores):
                    w = float(att_scores[idx])
                else:
                    w = 0.5
                idx += 1
                
                # 添加正向边
                KG[h].append((t, r_id, w))
                # 添加反向边（这对于路径搜索很重要）
                reverse_r_id = f"rev_{r_id}"
                KG[t].append((h, reverse_r_id, w))
        
        return KG
    
    KG = build_kg_with_attention(data_generator, att_scores)
    
    # 3. 保存路径
    save_dir = r"/content/drive/MyDrive/KGAT/Data/amazon-book"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "kgat_topk_paths.txt")
    
    # 4. 导出详细路径
    export_detailed_paths(sess, model, data_generator, KG, save_path)
    
    
    # 随机抽 50 个用户
    all_users = list(data_generator.train_user_dict.keys())
    np.random.seed(42)
    selected_users = list(np.random.choice(all_users, size=50, replace=False))

    # 训练阶段：每用户 2000 条路径
    train_path_file = os.path.join(save_dir, "train_50_users_2000_paths.txt")
    export_paths_for_users(sess, model, data_generator, KG,
                           save_path=train_path_file,
                           users_to_export=selected_users,
                           top_k_items=10,
                           max_paths_per_user=2000)

    # 测试阶段：每用户 20 条路径
    test_path_file = os.path.join(save_dir, "test_50_users_20_paths.txt")
    export_paths_for_users(sess, model, data_generator, KG,
                           save_path=test_path_file,
                           users_to_export=selected_users,
                           top_k_items=10,
                           max_paths_per_user=20)

    print("Done! All paths exported.")

    
    