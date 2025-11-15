import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import hnswlib

def squared_l2_distance(vec1, vec2):
    return sum((vec1[i] - vec2[i]) ** 2 for i in range(len(vec1)))

def linear(database, query, k):
    distances = [squared_l2_distance(vec, query[0]) for vec in database]
    sorted_indices = np.argsort(distances)[:k+1].tolist()
    return sorted_indices[1:k+1]

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
num_elements = len(database)

# EXPERIMENT: Vary k (fixed ef)
k_values = [1, 2, 3, 5, 10, 20, 50, 70, 100]
ef = 100
M = 16

# Build HNSW index 
p = hnswlib.Index(space='l2', dim=vector_dim)
p.init_index(max_elements=num_elements, ef_construction=ef, M=M)
start = time.time()
p.add_items(np.array(database))
p.set_ef(ef)
build_time_fixed = time.time() - start

prec_k, rec_k, mrr_k, map_k, ndcg_k = [], [], [], [], []
knn_times_k, hnsw_times_k = [], []

def compare_methods(database, query, k):
    start = time.time()
    knnResults = linear(database, query, k)
    knnTime = time.time() - start

    start = time.time()
    labels, _ = p.knn_query(np.array(query), k=k+1)
    hnswResults = list(labels[0][1:k+1])
    hnswTime = time.time() - start

    p1 = precision(knnResults, hnswResults, k)
    r1 = recall(knnResults, hnswResults, k)
    mrr1 = mean_reciprocal_rank(knnResults, hnswResults)
    ap1 = average_precision(knnResults, hnswResults)
    nd1 = ndcg(knnResults, hnswResults, k)
    return p1, r1, mrr1, ap1, nd1, knnTime, hnswTime

for k in k_values:
    p1, r1, mrr1, ap1, nd1, knnT, hnswT = compare_methods(database, query, k)
    prec_k.append(p1); rec_k.append(r1); mrr_k.append(mrr1); map_k.append(ap1); ndcg_k.append(nd1)
    knn_times_k.append(knnT); hnsw_times_k.append(hnswT)
print(np.round(hnsw_times_k,8))

results_k = pd.DataFrame({
    'k': k_values,
    'Precision': np.round(prec_k, 3),
    'Recall': np.round(rec_k, 3),
    'MRR': np.round(mrr_k, 3),
    'MAP': np.round(map_k, 3),
    'nDCG': np.round(ndcg_k, 3),
    'Linear Time (s)': np.round(knn_times_k, 5),
    'HNSW Time (s)': np.round(hnsw_times_k, 5),
    '× Faster (Linear/HNSW)': np.round(np.array(knn_times_k)/np.array(hnsw_times_k), 3)
})
print("\n" + "="*90)
print(f"(1) HNSW vs Linear — Varying k (ef = {ef}) | Build Time = {build_time_fixed} s")
print("="*90)
print(results_k.to_string(index=False))
print("="*90)

# EXPERIMENT: Vary ef (fixed k)
ef_values = [10, 20, 40, 60, 80, 100, 200]
k_fixed = 10

prec_e, rec_e, mrr_e, map_e, ndcg_e = [], [], [], [], []
knn_times_e, hnsw_times_e, build_times_e = [], [], []

for ef_value in ef_values:
    p = hnswlib.Index(space='l2', dim=vector_dim)
    p.init_index(max_elements=num_elements, ef_construction=ef_value, M=M)
    start = time.time()
    p.add_items(np.array(database))
    buildT = time.time() - start
    p.set_ef(ef_value)

    start = time.time()
    labels, _ = p.knn_query(np.array(query), k=k_fixed+1)
    hnswResults = list(labels[0][1:k_fixed+1])
    hnswTime = time.time() - start

    start = time.time()
    knnResults = linear(database, query, k_fixed)
    knnTime = time.time() - start

    p1 = precision(knnResults, hnswResults, k_fixed)
    r1 = recall(knnResults, hnswResults, k_fixed)
    mrr1 = mean_reciprocal_rank(knnResults, hnswResults)
    ap1 = average_precision(knnResults, hnswResults)
    nd1 = ndcg(knnResults, hnswResults, k_fixed)

    prec_e.append(p1); rec_e.append(r1); mrr_e.append(mrr1); map_e.append(ap1); ndcg_e.append(nd1)
    knn_times_e.append(knnTime); hnsw_times_e.append(hnswTime); build_times_e.append(buildT)

results_ef = pd.DataFrame({
    'ef': ef_values,
    'Precision': np.round(prec_e, 3),
    'Recall': np.round(rec_e, 3),
    'MRR': np.round(mrr_e, 3),
    'MAP': np.round(map_e, 3),
    'nDCG': np.round(ndcg_e, 3),
    'Build Time (s)': np.round(build_times_e, 4),
    'HNSW Query Time (s)': np.round(hnsw_times_e, 5),
    '× Faster (Linear/HNSW)': np.round(np.array(knn_times_e)/np.array(hnsw_times_e), 3)
})
print("\n" + "="*90)
print(f"(2) HNSW Performance — Varying ef (k = {k_fixed})")
print("="*90)
print(results_ef.to_string(index=False))
print("="*90)

# PLOTTING
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("HNSW vs Linear — Runtime and Ranking Performance", fontsize=18)

# Runtime vs k
axs[0,0].plot(k_values, [build_time_fixed] * len(k_values), label='Build Time', color='green')
axs[0,0].plot(k_values, knn_times_k, 'o-', color='red', label='Linear')
axs[0,0].plot(k_values, hnsw_times_k, 's-', color='blue', label=f'HNSW (ef={ef})')
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
axs[0,1].set_ylim(0.5, 1.05)
axs[0,1].legend(); axs[0,1].grid(True)

# Build + Query Time vs ef
axs[1,0].plot(ef_values, build_times_e, '^-', label='Build Time', color='green')
axs[1,0].plot(ef_values, knn_times_e, 'o-', color='red', label='Linear')
axs[1,0].plot(ef_values, hnsw_times_e, 's-', label='Query Time', color='blue')
axs[1,0].set_title("HNSW Runtime vs ef")
axs[1,0].set_xlabel("ef")
axs[1,0].set_ylabel("Time (s)")
axs[1,0].legend(); axs[1,0].grid(True)

# Ranking Metrics vs ef
axs[1,1].plot(ef_values, prec_e, '-', label='Precision', color='green')
axs[1,1].plot(ef_values, rec_e, '-.', label='Recall', color='orange')
axs[1,1].plot(ef_values, mrr_e, ':', label='MRR', color='purple')
axs[1,1].plot(ef_values, map_e, ':', label='MAP', color='blue')
axs[1,1].plot(ef_values, ndcg_e, ':', label='nDCG', color='red')
axs[1,1].set_title("Ranking Metrics vs ef")
axs[1,1].set_ylim(0.5, 1.05)
axs[1,1].legend(); axs[1,1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
