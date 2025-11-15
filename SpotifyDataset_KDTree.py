#source: https://www.geeksforgeeks.org/machine-learning/how-to-reduce-knn-computation-time-using-kd-tree-or-ball-tree/
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree, BallTree
from sklearn.metrics import pairwise_distances

def linear(database, query, k=5):
    squared_l2_distances = []
    for vec in database:
        squared_l2_distances.append(squared_l2_distance(vec, query[0]))
    sorted_indices = np.argsort(squared_l2_distances)[:k+1].tolist()
    top_k_similar = []
    for i in range(k):
        top_k_similar.append(sorted_indices[i+1])
    return top_k_similar

def squared_l2_distance(vec1, vec2):
    D = 0
    for i in range(len(vec1)):
        D += (vec1[i] - vec2[i])**2
    return D

# KD Tree Method
def kd_tree_query(database, query, k=5):
    tree = KDTree(database)
    distances, indices = tree.query(query, k+1)
    indices = indices.tolist()
    top_k_similar = []
    for i in range(k):
        top_k_similar.append(indices[0][i+1])
    return top_k_similar

#COMPARISON MEASURES
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
    score = 0.0
    num_hits = 0.0
    for i, doc_id in enumerate(approx_neighbors, start=1):
        if doc_id in true_neighbors:
            num_hits += 1
            score += num_hits / i
    return score / len(true_neighbors) if true_neighbors else 0.0

def ndcg(true_neighbors, approx_neighbors, k):
    idcg = 0.0
    for i in range(min(len(true_neighbors), k)):
        idcg += 1.0 / np.log2(i + 2)
    if idcg == 0:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(approx_neighbors[:k]):
        if doc_id in true_neighbors:
            dcg += 1.0 / np.log2(i + 2)
    return dcg / idcg

#COMPARE
def compare_methods(database, query, k=5):
    start = time.time()
    knnResult = linear(database, query, k)
    end = time.time()
    knnTime = end - start

    start = time.time()
    kdResult = kd_tree_query(database, query, k)
    end = time.time()
    kdTime = end - start

    P = precision(knnResult, kdResult, k)
    R = recall(knnResult, kdResult, k)
    MRR = mean_reciprocal_rank(knnResult, kdResult)
    MAP = average_precision(knnResult, kdResult)
    NDCG = ndcg(knnResult, kdResult, k)

    return P, R, MRR, MAP, NDCG, knnTime, kdTime

    

df = pd.read_csv('data.csv', index_col=0)

#selection of query and columns
queryIndex = 1936
#features selected:
columns = ['acousticness','danceability','energy','instrumentalness','liveness','speechiness','valence']

database=[]
query=[]#initialise
for i in range(len(df)):
    if i==queryIndex:
        query.append(df.loc[i,columns].values.tolist())
    database.append(df.loc[i,columns].values.tolist())         

np.random.seed(4200)

k_values = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
knn_times, kd_times= [], []
kd_precisions, kd_recalls = [], []
kd_mrrs, kd_maps, kd_ndcgs = [], [], []


num_runs = 50 

for k in k_values:
    avg_p = avg_r = avg_mrr = avg_map = avg_ndcg = avg_knn_time = avg_kd_time = 0.0

    for _ in range(num_runs):
        p, r, mrr, map, ndcg_val, knnTime, kdTime = compare_methods(database, query, k)
        avg_p += p
        avg_r += r
        avg_mrr += mrr
        avg_map += map
        avg_ndcg += ndcg_val
        avg_knn_time += knnTime
        avg_kd_time += kdTime

    # Take the average over the runs
    kd_precisions.append(avg_p / num_runs)
    kd_recalls.append(avg_r / num_runs)
    kd_mrrs.append(avg_mrr / num_runs)
    kd_maps.append(avg_map / num_runs)
    kd_ndcgs.append(avg_ndcg / num_runs)
    knn_times.append(avg_knn_time / num_runs)
    kd_times.append(avg_kd_time / num_runs)

print(kd_times)
print(f"{'k':<5}{'Precision':<12}{'Recall':<12}{'MRR':<12}{'MAP':<12}{'NDCG':<12}{'× Faster':<10}")
print("-" * 73)

for k, p, r, mrr, map, ndcg, tf in zip(
    k_values,
    np.round(kd_precisions, 3),
    np.round(kd_recalls, 3),
    np.round(kd_mrrs, 3),
    np.round(kd_maps, 3),
    np.round(kd_ndcgs, 3),
    np.round(np.array(knn_times) / np.array(kd_times), 3)
):
    print(f"{k:<5}{p:<12}{r:<12}{mrr:<12}{map:<12}{ndcg:<12}{tf:<10}")

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('KD Tree vs Linear: Runtime and Ranking Metrics', fontsize=16)

# Runtime
axs[0, 0].plot(k_values, knn_times, 'o-', label='Linear')
axs[0, 0].plot(k_values, kd_times, 's-', label='KD Tree')
axs[0, 0].set_title('Runtime vs K')
axs[0, 0].set_xlabel('K')
axs[0, 0].set_ylabel('Seconds')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Precision / Recall
axs[0, 1].plot(k_values, kd_precisions, 's-', label='Precision')
axs[0, 1].plot(k_values, kd_recalls, 'x-', label='Recall')
axs[0, 1].set_title('Precision & Recall vs K')
axs[0, 1].set_xlabel('K')
axs[0, 1].set_ylabel('Score')
axs[0, 1].legend()
axs[0, 1].grid(True)

# MRR / MAP
axs[1, 0].plot(k_values, kd_mrrs, 's-', label='MRR')
axs[1, 0].plot(k_values, kd_maps, 'x-', label='MAP')
axs[1, 0].set_title('MRR & MAP vs K')
axs[1, 0].set_xlabel('K')
axs[1, 0].set_ylabel('Score')
axs[1, 0].legend()
axs[1, 0].grid(True)

# nDCG
axs[1, 1].plot(k_values, kd_ndcgs, 'd-', label='nDCG')
axs[1, 1].set_title('nDCG vs K')
axs[1, 1].set_xlabel('K')
axs[1, 1].set_ylabel('Score')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()