import re
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

########################################
# 配置
########################################
BASE_DIR = r"D:\Thesis_Project\Models\KGAT\Data\amazon-beauty"

PATH_FILE = f"{BASE_DIR}/kgat_topk_paths.txt"
ENTITY_FILE = f"{BASE_DIR}/entity2global_id.txt"
TIME_TRAIN_FILE = f"{BASE_DIR}/time_train.csv"
TIME_TEST_FILE = f"{BASE_DIR}/time_test.csv"
KG_FILE = f"{BASE_DIR}/kg_final.txt"

TOP_K = 10
BETA_LIR = 0.5
BETA_SEP = 0.5

########################################
# Step 1: 解析路径文件
########################################
print("[INFO] Loading KGAT paths from txt...")

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

        elif line.startswith("(") and "," in line and "attention" in line:
            # (h, r, t), attention=...
            triple = re.search(r"\((\d+),\s*([^,]+),\s*(\d+)\)", line)
            if triple:
                h = int(triple.group(1))
                r = triple.group(2)
                t = int(triple.group(3))
                current_path.append((h, r, t))

        elif line == "" and current_path:
            if current_user is not None:
                user_paths[current_user].append(current_path)
            current_path = []

# 截断 Top-K
for u in user_paths:
    user_paths[u] = user_paths[u][:TOP_K]

print(f"[INFO] Parsed paths for {len(user_paths)} users")

########################################
# Step 2: 计算 KG 全局 in-degree（SEP 用）
########################################
print("[INFO] Computing global entity indegree from KG...")

entity_indegree = defaultdict(int)

with open(KG_FILE, "r") as f:
    for line in f:
        h, r, t = map(int, line.strip().split())
        entity_indegree[t] += 1

deg_vals = list(entity_indegree.values())
min_deg, max_deg = min(deg_vals), max(deg_vals)

entity_popularity = {
    eid: (deg - min_deg) / (max_deg - min_deg) if max_deg > min_deg else 0.0
    for eid, deg in entity_indegree.items()
}

########################################
# Step 3: 加载用户交互时间（LIR 用）
########################################
print("[INFO] Loading interaction timestamps...")

train_df = pd.read_csv(TIME_TRAIN_FILE)
test_df = pd.read_csv(TIME_TEST_FILE)
df = pd.concat([train_df, test_df])

df["PURCHASE_Time"] = pd.to_datetime(df["PURCHASE_Time"], errors="coerce")
df = df.dropna()

min_date = df["PURCHASE_Time"].min()
df["t_days"] = (df["PURCHASE_Time"] - min_date).dt.days

user_item_time = defaultdict(dict)
for row in df.itertuples(index=False):
    user_item_time[row.UID][row.PID] = row.t_days

########################################
# Step 4: LIR
########################################
def compute_lir(uid, path):
    times = []
    for h, r, t in path:
        if t in user_item_time.get(uid, {}):
            times.append(user_item_time[uid][t])

    if not times:
        return None

    times.sort()
    lir = times[0]
    for x in times[1:]:
        lir = (1 - BETA_LIR) * lir + BETA_LIR * x
    return lir

user_lir = {}
for uid, paths in tqdm(user_paths.items(), desc="Computing LIR"):
    vals = [compute_lir(uid, p) for p in paths]
    vals = [v for v in vals if v is not None]

    if vals:
        mn, mx = min(vals), max(vals)
        user_lir[uid] = 0.0 if mn == mx else np.mean([(x - mn) / (mx - mn) for x in vals])

########################################
# Step 5: SEP（所有非 user 实体）
########################################
def compute_sep(path):
    pops = [entity_popularity.get(t, 0.0) for _, _, t in path]
    if not pops:
        return None

    sep = pops[0]
    for v in pops[1:]:
        sep = (1 - BETA_SEP) * sep + BETA_SEP * v
    return sep

user_sep = {}
for uid, paths in tqdm(user_paths.items(), desc="Computing SEP"):
    vals = [compute_sep(p) for p in paths]
    vals = [v for v in vals if v is not None]
    if vals:
        user_sep[uid] = np.mean(vals)

########################################
# Step 6: ETD（关系 pattern 多样性）
########################################
global_patterns = set()
for paths in user_paths.values():
    for p in paths:
        pattern = tuple(r for _, r, _ in p)
        if pattern:
            global_patterns.add(pattern)

G = len(global_patterns)

user_etd = {}
for uid, paths in tqdm(user_paths.items(), desc="Computing ETD"):
    patterns = set(tuple(r for _, r, _ in p) for p in paths if p)
    if patterns:
        user_etd[uid] = len(patterns) / min(TOP_K, G)

########################################
# Step 7: 输出结果
########################################
print("\n========== FINAL RESULTS ==========")
print(f"LIR: {np.mean(list(user_lir.values())):.4f}")
print(f"SEP: {np.mean(list(user_sep.values())):.4f}")
print(f"ETD: {np.mean(list(user_etd.values())):.4f}")
