import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from annoy import AnnoyIndex

def squared_l2_distance(vec1, vec2):
    return sum((vec1[i] - vec2[i]) ** 2 for i in range(len(vec1)))

def linear(database, query, k):
    distances = [squared_l2_distance(vec, query[0]) for vec in database]
    sorted_indices = np.argsort(distances)[:k+1].tolist()
    return sorted_indices[1:k+1]  # skip the query itself

def annoy_query(annoy_index, query, k):
    results = annoy_index.get_nns_by_vector(query[0], k+1, include_distances=False)
    return results[1:k+1]  # skip the query itself

# Evaluation Metrics 
def precision(true_neighbors, approx_neighbors, k):
    intersection = len(set(true_neighbors) & set(approx_neighbors))
    return intersection / len(approx_neighbors) if approx_neighbors else 0

def recall(true_neighbors, approx_neighbors, k):
    intersection = len(set(true_neighbors) & set(approx_neighbors))
    return intersection / len(true_neighbors) if true_neighbors else 0

def mean_reciprocal_rank(true_neighbors, approx_neighbors):
    for rank, doc_id in enumerate(approx_neighbors, start=1):
        if doc_id in true_neighbors:
            return 1.0 / rank
    return 0.0

def average_precision(true_neighbors, approx_neighbors):
    score, num_hits = 0.0, 0.0
    for i, doc_id in enumerate(approx_neighbors, start=1):
        if doc_id in true_neighbors:
            num_hits += 1
            score += num_hits / i
    return score / len(true_neighbors) if true_neighbors else 0.0

def ndcg(true_neighbors, approx_neighbors, k):
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_neighbors), k)))
    if idcg == 0: return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, doc_id in enumerate(approx_neighbors[:k]) if doc_id in true_neighbors)
    return dcg / idcg

# Load Dataset 
df = pd.read_csv('data.csv', index_col=0)
columns = ['acousticness','danceability','energy','instrumentalness','liveness','speechiness','valence']
queryIndex = 1936

database = [df.loc[i, columns].values.tolist() for i in range(len(df))]
query = [df.loc[queryIndex, columns].values.tolist()]
vector_dim = len(columns)

# Comparison Function
def compare_methods(database, query, annoy_index, k):
    start = time.time()
    knnResults = linear(database, query, k)
    knnTime = time.time() - start

    start = time.time()
    annResults = annoy_query(annoy_index, query, k)
    annTime = time.time() - start

    p = precision(knnResults, annResults, k)
    r = recall(knnResults, annResults, k)
    mrr = mean_reciprocal_rank(knnResults, annResults)
    ap = average_precision(knnResults, annResults)
    nd = ndcg(knnResults, annResults, k)
    return p, r, mrr, ap, nd, knnTime, annTime


# EXPERIMENT: Vary k (fixed trees)
num_trees = 10
k_values = [1, 2, 3, 5, 10, 20, 50, 70, 100]


ann_index = AnnoyIndex(vector_dim, 'euclidean')
for i, vec in enumerate(database):
    ann_index.add_item(i, vec)
start = time.time()
ann_index.build(num_trees)
build_time_fixed_trees = time.time() - start

prec_k, rec_k, mrr_k, map_k, ndcg_k = [], [], [], [], []
knn_times_k, annoy_times_k = [], []

num_runs = 1000
for k in k_values:
    avg_p = avg_r = avg_mrr = avg_ap = avg_nd = avg_knnT = avg_annT = 0.0

    for _ in range(num_runs):
        p, r, mrr, ap, nd, knnT, annT = compare_methods(database, query, ann_index, k)
        avg_p += p
        avg_r += r
        avg_mrr += mrr
        avg_ap += ap
        avg_nd += nd
        avg_knnT += knnT
        avg_annT += annT

    prec_k.append(avg_p / num_runs)
    rec_k.append(avg_r / num_runs)
    mrr_k.append(avg_mrr / num_runs)
    map_k.append(avg_ap / num_runs)
    ndcg_k.append(avg_nd / num_runs)
    knn_times_k.append(avg_knnT / num_runs)
    annoy_times_k.append(avg_annT / num_runs)

print(np.round(knn_times_k,6))
print(np.round(annoy_times_k,8))

# --- Table for varying k ---
results_k = pd.DataFrame({
    'k': k_values,
    'Precision': np.round(prec_k, 3),
    'Recall': np.round(rec_k, 3),
    'MRR': np.round(mrr_k, 3),
    'MAP': np.round(map_k, 3),
    'nDCG': np.round(ndcg_k, 3),
    'Linear Time (s)': np.round(knn_times_k, 5),
    'Annoy Time (s)': np.round(annoy_times_k, 5),
    '× Faster (Linear/Annoy)': np.round(np.array(knn_times_k)/np.array(annoy_times_k), 3)
})

print("\n" + "="*90)
print(f"(1) Annoy vs Linear — Varying k (num_trees = {num_trees}) | Build Time = {build_time_fixed_trees} s")
print("="*90)
print(results_k.to_string(index=False))
print("="*90)


# EXPERIMENT: Vary number of trees (fixed k)
trees_values = [1, 2, 5, 10, 20, 40, 60, 80, 100]
k_fixed = 10

prec_t, rec_t, mrr_t, map_t, ndcg_t = [], [], [], [], []
knn_times_t, annoy_times_t, build_times_t = [], [], []

for n_trees in trees_values:
    ann_index = AnnoyIndex(vector_dim, 'euclidean')
    for i, vec in enumerate(database):
        ann_index.add_item(i, vec)

    start = time.time()
    ann_index.build(n_trees)
    buildT = time.time() - start

    p, r, mrr, ap, nd, knnT, annT = compare_methods(database, query, ann_index, k_fixed)
    prec_t.append(p); rec_t.append(r); mrr_t.append(mrr); map_t.append(ap); ndcg_t.append(nd)
    knn_times_t.append(knnT); annoy_times_t.append(annT); build_times_t.append(buildT)

# Table for varying num_trees
results_trees = pd.DataFrame({
    '# Trees': trees_values,
    'Precision': np.round(prec_t, 3),
    'Recall': np.round(rec_t, 3),
    'MRR': np.round(mrr_t, 3),
    'MAP': np.round(map_t, 3),
    'nDCG': np.round(ndcg_t, 3),
    'Build Time (s)': np.round(build_times_t, 4),
    'Annoy Query Time (s)': np.round(annoy_times_t, 5),
    '× Faster (Linear/Annoy)': np.round(np.array(knn_times_t)/np.array(annoy_times_t), 3)
})

print("\n" + "="*90)
print(f"(2) Annoy Performance — Varying Number of Trees (k = {k_fixed})")
print("="*90)
print(results_trees.to_string(index=False))
print("="*90)


# PLOTTING
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Annoy vs Linear — Runtime and Ranking Performance", fontsize=18)

# Runtime vs k
axs[0,0].plot(k_values, knn_times_k, 'o-', color='red', label='Linear')
axs[0,0].plot(k_values, annoy_times_k, 's-', color='blue', label=f'Annoy ({num_trees} trees)')
axs[0,0].set_title("Query Runtime vs k")
axs[0,0].set_xlabel("k (Neighbors)")
axs[0,0].set_ylabel("Runtime (s)")
axs[0,0].legend(); axs[0,0].grid(True)

# Ranking Metrics vs k 
axs[0,1].plot(k_values, prec_k, '-', label='Precision', color='green')
axs[0,1].plot(k_values, rec_k, '-.', label='Recall', color='orange')
axs[0,1].plot(k_values, mrr_k, ':', label='MRR', color='purple')
axs[0,1].plot(k_values, map_k, ':', label='MAP', color='blue')
axs[0,1].plot(k_values, ndcg_k, ':', label='nDCG', color='red')
axs[0,1].set_title("Ranking Metrics vs k")
axs[0,1].set_xlabel("k (Neighbors)")
axs[0,1].set_ylim(0.5, 1.05)
axs[0,1].legend(); axs[0,1].grid(True)

# Build + Query Time vs Trees 
axs[1,0].plot(trees_values, build_times_t, '^-', label='Build Time', color='green')
axs[1,0].plot(trees_values, annoy_times_t, 's-', label='Query Time', color='blue')
axs[1,0].set_title("Annoy Runtime vs Number of Trees")
axs[1,0].set_xlabel("Number of Trees")
axs[1,0].set_ylabel("Time (s)")
axs[1,0].legend(); axs[1,0].grid(True)

# Ranking Metrics vs Trees 
axs[1,1].plot(trees_values, prec_t, '-', label='Precision', color='green')
axs[1,1].plot(trees_values, rec_t, '-.', label='Recall', color='orange')
axs[1,1].plot(trees_values, mrr_t, ':', label='MRR', color='purple')
axs[1,1].plot(trees_values, map_t, ':', label='MAP', color='blue')
axs[1,1].plot(trees_values, ndcg_t, ':', label='nDCG', color='red')
axs[1,1].set_title("Ranking Metrics vs Number of Trees")
axs[1,1].set_xlabel("Number of Trees")
axs[1,1].set_ylim(0.5, 1.05)
axs[1,1].legend(); axs[1,1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
