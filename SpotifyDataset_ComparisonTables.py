import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

k_values = [1, 5, 10, 20, 50, 100] 

annoy_time = [0.000005750, 0.000009300, 0.000001316, 0.000002098, 0.000004053, 0.000007202]
annoy_p = [1,0.4,0.9,1,0.98,1]
annoy_r = [1,0.4,0.9,1,0.98,1]
annoy_mrr = [1,1,1,1,1,1]
annoy_map = [1,0.4,0.9,1,0.98,1]
annoy_ndcg = [1,0.553,0.936,1,0.986,1]

balltree_time = [0.0008900, 0.0007899, 0.0007911, 0.0008028, 0.0007808, 0.0007699]
balltree_p = [1,1,1,1,1,1]
balltree_r = [1,1,1,1,1,1]
balltree_mrr = [1,1,1,1,1,1]
balltree_map = [1,1,1,1,1,1]
balltree_ndcg = [1,1,1,1,1,1]

lsh_time = [0.0006574, 0.0007052, 0.0007042, 0.0006755, 0.0006685, 0.0006544]
lsh_p = [1,0.8,0.9,0.8,0.68,0.530]
lsh_r = [1,0.8,0.9,0.8,0.68,0.530]
lsh_mrr = [1,1,1,1,1,1]
lsh_map = [1,0.8,0.9,0.8,0.68,0.530]
lsh_ndcg = [1,0.869,0.936,0.867,0.772,0.641]

knn_time = [0.002557, 0.002597, 0.002617, 0.002593, 0.00258, 0.002652]
knn_p = [1,1,1,1,1,1]
knn_r = [1,1,1,1,1,1]
knn_mrr = [1,1,1,1,1,1]
knn_map = [1,1,1,1,1,1]
knn_ndcg = [1,1,1,1,1,1]

hnsw_time = [0.00004983, 0.00003409, 0.00003004, 0.00002575, 0.00002694, 0.00003099]
hnsw_p = [1,1,1,1,1,1]
hnsw_r = [1,1,1,1,1,1]
hnsw_mrr = [1,1,1,1,1,1]
hnsw_map = [1,1,1,1,1,1]
hnsw_ndcg = [1,1,1,1,1,1]

kdtree_time = [0.0008497, 0.0007999, 0.0007746, 0.0007879, 0.0009365, 0.0007515]
kdtree_p = [1,1,1,1,1,1]
kdtree_r = [1,1,1,1,1,1]
kdtree_mrr = [1,1,1,1,1,1]
kdtree_map = [1,1,1,1,1,1]
kdtree_ndcg = [1,1,1,1,1,1]

# Plot

# Average Query Time per Query vs k
plt.figure(figsize=(8, 5))
plt.plot(k_values, annoy_time, 'o-', label='Annoy')
plt.plot(k_values, balltree_time, 'o-', label='BallTree')
plt.plot(k_values, lsh_time, 'o-', label='E2LSH')
plt.plot(k_values, knn_time, 'o-', label='ExactBrute')
plt.plot(k_values, hnsw_time, 'o-', label='HNSW')
plt.plot(k_values, kdtree_time, 'o-', label='KDTree')

plt.xlabel('k (Neighbours)')
plt.ylabel('Time (s)')
plt.title('Avg Query Time per Query vs k')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#Precision@k for each method
data = {
    "BallTree": balltree_p,
    "ExactBrute": knn_p,  
    "KDTree": kdtree_p,
    "HNSW": hnsw_p,
    "Annoy": annoy_p,
    "E2LSH": lsh_p
}
df = pd.DataFrame(data, index=k_values).T  # transpose so methods are rows
df["mean"] = df.mean(axis=1)

plt.figure(figsize=(10, 5))
sns.heatmap(df, annot=True, cmap="Blues_r", fmt=".3f", linewidths=0.5, cbar=False)

plt.title("Precision@k by Method", fontsize=14, weight="bold")
plt.xlabel("k")
plt.ylabel("method")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Recall@k for each method
data = {
    "BallTree": balltree_r,
    "ExactBrute": knn_r,  
    "KDTree": kdtree_r,
    "HNSW": hnsw_r,
    "Annoy": annoy_r,
    "E2LSH": lsh_r
}
df = pd.DataFrame(data, index=k_values).T  # transpose so methods are rows
df["mean"] = df.mean(axis=1)

plt.figure(figsize=(10, 5))
sns.heatmap(df, annot=True, cmap="Blues_r", fmt=".3f", linewidths=0.5, cbar=False)

plt.title("Recall@k by Method", fontsize=14, weight="bold")
plt.xlabel("k")
plt.ylabel("method")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# MRR@k for each method
data = {
    "BallTree": balltree_mrr,
    "ExactBrute": knn_mrr,  
    "KDTree": kdtree_mrr,
    "HNSW": hnsw_mrr,
    "Annoy": annoy_mrr,
    "E2LSH": lsh_mrr
}
df = pd.DataFrame(data, index=k_values).T  
df["mean"] = df.mean(axis=1)

plt.figure(figsize=(10, 5))
sns.heatmap(df, annot=True, cmap="Blues_r", fmt=".3f", linewidths=0.5, cbar=False)

plt.title("MRR@k by Method", fontsize=14, weight="bold")
plt.xlabel("k")
plt.ylabel("method")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#MAP@k for each method
data = {
    "BallTree": balltree_map,
    "ExactBrute": knn_map,  
    "KDTree": kdtree_map,
    "HNSW": hnsw_map,
    "Annoy": annoy_map,
    "E2LSH": lsh_map
}
df = pd.DataFrame(data, index=k_values).T 
df["mean"] = df.mean(axis=1)

plt.figure(figsize=(10, 5))
sns.heatmap(df, annot=True, cmap="Blues_r", fmt=".3f", linewidths=0.5, cbar=False)

plt.title("MAP@k by Method", fontsize=14, weight="bold")
plt.xlabel("k")
plt.ylabel("method")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# nDCG@k for each method
data = {
    "BallTree": balltree_ndcg,
    "ExactBrute": knn_ndcg,  
    "KDTree": kdtree_ndcg,
    "HNSW": hnsw_ndcg,
    "Annoy": annoy_ndcg,
    "E2LSH": lsh_ndcg
}
df = pd.DataFrame(data, index=k_values).T  
df["mean"] = df.mean(axis=1)

plt.figure(figsize=(10, 5))
sns.heatmap(df, annot=True, cmap="Blues_r", fmt=".3f", linewidths=0.5, cbar=False)

plt.title("nDCG@k by Method", fontsize=14, weight="bold")
plt.xlabel("k")
plt.ylabel("method")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()