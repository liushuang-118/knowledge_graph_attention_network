import re
import numpy as np
from collections import defaultdict, Counter
from scipy.spatial.distance import jensenshannon

# ===============================
# 配置
# ===============================
BASE_DIR = r"D:\Thesis_Project\Models\KGAT\Data\amazon-beauty"

TRAIN_PATH_FILE = f"{BASE_DIR}/train_50_users_2000_paths.txt"
TEST_PATH_FILE  = f"{BASE_DIR}/test_50_users_20_paths.txt"

EPS = 1e-12

# ===============================
# Step 1: 解析 KGAT path 文件
# ===============================
def load_user_relation_counts(path_file, weighted=False):
    """
    weighted = False → F(u), Qf(u)  (frequency)
    weighted = True  → Qw(u)        (attention-weighted)
    """
    user_rel_counter = defaultdict(Counter)

    current_user = None
    current_rels = []   # [(rel, attention)]

    with open(path_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # User header
            if line.startswith("User"):
                current_user = int(line.split()[1])

            # Path header
            elif line.startswith("Path"):
                current_rels = []

            # Triple with attention
            elif line.startswith("(") and "attention" in line:
                triple = re.search(
                    r"\(\d+,\s*([^,]+),\s*\d+\).*attention=([\d\.eE+-]+)",
                    line
                )
                if triple:
                    rel = triple.group(1)
                    att = float(triple.group(2))
                    current_rels.append((rel, att))

            # End of one path
            elif line == "" and current_user is not None and current_rels:
                for rel, att in current_rels:
                    if weighted:
                        user_rel_counter[current_user][rel] += att
                    else:
                        user_rel_counter[current_user][rel] += 1
                current_rels = []

    return user_rel_counter


def normalize(counter_dict):
    norm = {}
    for uid, counter in counter_dict.items():
        total = sum(counter.values())
        if total > 0:
            norm[uid] = {k: v / total for k, v in counter.items()}
    return norm


# ===============================
# Step 2: 构建 F(u), Qf(u), Qw(u)
# ===============================
print("[INFO] Loading training paths (F(u))...")
F_u_raw = load_user_relation_counts(TRAIN_PATH_FILE, weighted=False)
F_u = normalize(F_u_raw)

print("[INFO] Loading test paths (Qf(u), Qw(u))...")
Qf_u_raw = load_user_relation_counts(TEST_PATH_FILE, weighted=False)
Qw_u_raw = load_user_relation_counts(TEST_PATH_FILE, weighted=True)

Qf_u = normalize(Qf_u_raw)
Qw_u = normalize(Qw_u_raw)

common_users = set(F_u.keys()) & set(Qf_u.keys())
print(f"[INFO] Common users: {len(common_users)}")

# ===============================
# Step 3: JS divergence
# ===============================
def js_divergence(p, q):
    keys = set(p.keys()) | set(q.keys())
    p_vec = np.array([p.get(k, 0.0) + EPS for k in keys])
    q_vec = np.array([q.get(k, 0.0) + EPS for k in keys])
    p_vec /= p_vec.sum()
    q_vec /= q_vec.sum()
    return jensenshannon(p_vec, q_vec, base=2.0) ** 2


jsf_list, jsw_list = [], []

for uid in common_users:
    jsf_list.append(js_divergence(Qf_u[uid], F_u[uid]))
    jsw_list.append(js_divergence(Qw_u[uid], F_u[uid]))

JSf = np.mean(jsf_list)
JSw = np.mean(jsw_list)

# ===============================
# Step 4: 输出
# ===============================
print("========== Explanation Faithfulness ==========")
print(f"[INFO] JS_f (frequency) : {JSf:.6f}")
print(f"[INFO] JS_w (weighted)  : {JSw:.6f}")
print(f"[INFO] #users           : {len(jsf_list)}")
