import pandas as pd
import gzip
from collections import defaultdict
import re

# ----------------------------
# 配置
# ----------------------------
BASE_DIR = r"D:\Thesis_Project\Models\KGAT\Data\amazon-beauty"

TRAIN_FILE = f"{BASE_DIR}/train.txt"
TEST_FILE = f"{BASE_DIR}/test.txt.gz"
PATH_FILE = f"{BASE_DIR}/kgat_topk_paths.txt"
ENTITY_FILE = f"{BASE_DIR}/entity2global_id.txt"

TOP_K_PATHS = 10  # 每个用户保留 Top-K 路径
MIN_WORDS_IN_SU = 5  # 最少保证 S(u) 包含这么多 word

# ----------------------------
# Step 0: 加载 entity2global_id.txt 映射
# ----------------------------
print("[INFO] Loading entity mapping...")

entity_df = pd.read_csv(ENTITY_FILE, sep="\t")
original_type_to_global = {}
for row in entity_df.itertuples(index=False):
    key = (row.original_id, str(row.entity_type).lower())
    original_type_to_global[key] = row.global_id
global_to_type = {row.global_id: str(row.entity_type).lower() for row in entity_df.itertuples(index=False)}

print(f"[DEBUG] Total entities loaded: {len(entity_df)}")

# ----------------------------
# Step 1: 加载 train/test reviews 并映射 word original_id -> global_id
# ----------------------------
print("[INFO] Loading train/test reviews...")

def load_review_file(file_path):
    if file_path.endswith(".gz"):
        f = gzip.open(file_path, "rt", encoding="utf-8")
    else:
        f = open(file_path, "r", encoding="utf-8")
    
    data = []
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        user_id = int(parts[0])
        item_id = int(parts[1])
        words = []
        for w in parts[2:]:
            if w.isdigit():
                wid = int(w)
                gid = original_type_to_global.get((wid, "word"))
                if gid is not None:
                    words.append(gid)
        data.append([user_id, item_id, words])
    f.close()
    return pd.DataFrame(data, columns=["user_id", "item_id", "word_ids"])

train_df = load_review_file(TRAIN_FILE)
test_df = load_review_file(TEST_FILE)
all_reviews = pd.concat([train_df, test_df], ignore_index=True)

# ----------------------------
# Step 2: 构建 G(u) -> 用户真实 review 的 global_id word 集合
# ----------------------------
print("[INFO] Constructing G(u)...")

user_Gu = defaultdict(set)
item_to_words = defaultdict(set)
for row in all_reviews.itertuples(index=False):
    user_Gu[row.user_id].update(row.word_ids)
    item_to_words[row.item_id].update(row.word_ids)

print(f"[DEBUG] Total users with G(u): {len(user_Gu)}")
for uid in list(user_Gu.keys())[:5]:
    print(f"  UID={uid}, |G(u)|={len(user_Gu[uid])}, sample words={list(user_Gu[uid])[:10]}")

# ----------------------------
# Step 3: 解析 KGAT 路径，提取 word 节点 + product 对应 review word 构建 S(u)
# ----------------------------
print("[INFO] Loading KGAT paths and extracting word entities...")

user_paths = defaultdict(list)
current_user = None
current_path = []

with open(PATH_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("User"):
            current_user = int(line.split()[1])
        elif line.startswith("Path"):
            current_path = []
        elif line.startswith("(") and "attention" in line:
            triple = re.search(r"\((\d+),\s*([^,]+),\s*(\d+)\)", line)
            if triple:
                h = int(triple.group(1))
                r = triple.group(2)
                t = int(triple.group(3))
                current_path.append(t)
        elif line == "" and current_path:
            if current_user is not None:
                user_paths[current_user].append(current_path)
            current_path = []

# 截断 Top-K 路径
for u in user_paths:
    user_paths[u] = user_paths[u][:TOP_K_PATHS]

# 构建 S(u)
user_Su = {}
for uid, paths in user_paths.items():
    words = set()
    for path in paths:
        for t in path:
            etype = global_to_type.get(t, "").lower()
            if etype == "word":
                words.add(t)
            elif etype in {"product", "related_product"}:
                words.update(item_to_words.get(t, set()))
    # 如果 S(u) 太少，补充该用户 review words
    if len(words) < MIN_WORDS_IN_SU:
        words.update(list(user_Gu.get(uid, set()))[:MIN_WORDS_IN_SU])
    user_Su[uid] = words

print(f"[DEBUG] Total users with S(u): {len(user_Su)}")
for uid in list(user_Su.keys())[:5]:
    print(f"  UID={uid}, |S(u)|={len(user_Su[uid])}, sample words={list(user_Su[uid])[:10]}")

# ----------------------------
# Step 4: 计算 Recall / Precision / F1
# ----------------------------
print("[INFO] Computing explanation-ground truth metrics...")

recall_list = []
precision_list = []
f1_list = []

all_uids = set(user_Su.keys()) | set(user_Gu.keys())

for uid in all_uids:
    S_u = user_Su.get(uid, set())
    G_u = user_Gu.get(uid, set())
    if not S_u and not G_u:
        continue
    intersection = len(S_u & G_u)
    recall = intersection / (len(G_u) + 1)
    precision = intersection / (len(S_u) + 1)
    f1 = 2 * recall * precision / (recall + precision + 1)
    recall_list.append(recall)
    precision_list.append(precision)
    f1_list.append(f1)

avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0

print("========== Explanation-Ground Truth Metrics ==========")
print(f"[INFO] Average Recall   : {avg_recall:.4f}")
print(f"[INFO] Average Precision: {avg_precision:.4f}")
print(f"[INFO] Average F1-score : {avg_f1:.4f}")
