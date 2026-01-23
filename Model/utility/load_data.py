'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import collections
import numpy as np
import random as rd

class Data(object):
    def __init__(self, args, path):
        self.path = path
        self.args = args

        self.batch_size = args.batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        kg_file = path + '/kg_final.txt'

        # ----------get number of users and items & then load rating data from train_file & test_file------------.
        self.n_train, self.n_test = 0, 0
        self.n_users, self.n_items = 0, 0

        self.train_data, self.train_user_dict = self._load_ratings(train_file)
        self.test_data, self.test_user_dict = self._load_ratings(test_file)
        self.exist_users = self.train_user_dict.keys()

        self._statistic_ratings()

        # ----------get number of entities and relations & then load kg data from kg_file ------------.
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)

        # ----------print the basic info about the dataset-------------.
        self.batch_size_kg = self.n_triples // (self.n_train // self.batch_size)
        self._print_data_info()

    # reading train & test interaction data.
    def _load_ratings(self, file_name):
        user_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                user_dict[u_id] = pos_ids
        return np.array(inter_mat), user_dict

    def _statistic_ratings(self):
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

    # reading train & test interaction data.
    def _load_kg(self, file_name):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        # self.n_relations = len(set(kg_np[:, 1]))
        # self.n_entities = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        self.n_triples = len(kg_np)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def _print_data_info(self):
        print('[n_users, n_items]=[%d, %d]' % (self.n_users, self.n_items))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_entities, n_relations, n_triples]=[%d, %d, %d]' % (self.n_entities, self.n_relations, self.n_triples))
        print('[batch_size, batch_size_kg]=[%d, %d]' % (self.batch_size, self.batch_size_kg))

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(list(self.exist_users), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    # 在 load_data.py 的 Data 类中添加：
    def generate_train_batch(self):
        """生成CF训练批次数据"""
        return self._generate_train_cf_batch()
    
    def generate_train_feed_dict(self, model, batch_data):
        """生成CF训练feed_dict"""
        users, pos_items, neg_items = batch_data
        feed_dict = {
            model.users: users,
            model.pos_items: pos_items,
            model.neg_items: neg_items,
            model.node_dropout: eval(self.args.node_dropout),
            model.mess_dropout: eval(self.args.mess_dropout)
        }
        return feed_dict
    
    def generate_train_A_batch(self):
        """生成KG训练批次数据"""
        exist_heads = list(self.all_kg_dict.keys())
        if self.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.batch_size_kg)]
        
        relations, pos_tails, neg_tails = [], [], []
        for h in heads:
            t_r_list = self.all_kg_dict[h]
            index = np.random.randint(low=0, high=len(t_r_list), size=1)[0]
            t, r = t_r_list[index]
            relations.append(r)
            pos_tails.append(t)
            
            # 采样负尾实体
            while True:
                neg_t = np.random.randint(low=0, high=self.n_entities, size=1)[0]
                if (neg_t, r) not in t_r_list:
                    neg_tails.append(neg_t)
                    break
        
        return heads, relations, pos_tails, neg_tails

    def generate_train_A_feed_dict(self, model, A_batch_data):
        """生成KG训练feed_dict"""
        heads, relations, pos_tails, neg_tails = A_batch_data
        feed_dict = {
            model.h: heads,
            model.r: relations,
            model.pos_t: pos_tails,
            model.neg_t: neg_tails,
            model.node_dropout: eval(self.args.node_dropout),
            model.mess_dropout: eval(self.args.mess_dropout)
        }
        return feed_dict
        
    def generate_test_feed_dict(self, model, user_batch, item_batch, drop_flag=False):
        """生成测试feed_dict"""
        feed_dict = {
            model.users: user_batch,
            model.pos_items: item_batch,
            model.node_dropout: [0.] * len(eval(self.args.layer_size)),
            model.mess_dropout: [0.] * len(eval(self.args.layer_size))
        }
        return feed_dict
    
    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_user_dict[uid]
            test_iids = self.test_user_dict[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)


        return split_uids, split_state