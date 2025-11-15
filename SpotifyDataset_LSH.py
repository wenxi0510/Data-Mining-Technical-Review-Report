# Import the needs
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt

import json
import time

'''
# Load dataset
dfSongs = pd.read_csv('data.csv', index_col=0)

# Number of rows and columns
rows, cols = dfSongs.shape
print('Number of songs: {}'.format(rows))
Number of songs: 2017
print('Number of attributes per song: {}'.format(cols))
Number of attributes per song: 16

# Print the columns
print(dfSongs.columns)
Number of attributes per song: 16
Index(['acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'time_signature', 'valence', 'target',
       'song_title', 'artist'],
      dtype='object')
<class 'pandas.core.frame.DataFrame'>
Index: 2017 entries, 0 to 2016

# Print the attributes type
dfSongs.info()
Data columns (total 16 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   acousticness      2017 non-null   float64
 1   danceability      2017 non-null   float64
 2   duration_ms       2017 non-null   int64  
 3   energy            2017 non-null   float64
 4   instrumentalness  2017 non-null   float64
 5   key               2017 non-null   int64  
 6   liveness          2017 non-null   float64
 7   loudness          2017 non-null   float64
 8   mode              2017 non-null   int64  
 9   speechiness       2017 non-null   float64
 10  tempo             2017 non-null   float64
 11  time_signature    2017 non-null   float64
 12  valence           2017 non-null   float64
 13  target            2017 non-null   int64  
 14  song_title        2017 non-null   object 
 15  artist            2017 non-null   object 
dtypes: float64(10), int64(4), object(2)
memory usage: 267.9+ KB'''

def getMusicName(song):
    return song['artist'] + " - " + song['song_title'] 

# IMPLEMENTING LSH
def linear(query, dataset, k=1):
    # query: dictionary with 1 vector {"...":[...]}
    # dataset: dictionary of vetors {"...":[...], "...":[...], ...}

    query_key = list(query.keys())[0]
    dataset_keys = list(dataset.keys())
    squared_l2_distances = {}
    for key in dataset_keys:
        squared_l2_distances[key] = squared_l2_distance(query[query_key], dataset[key])
    squared_l2_distances = sorted(squared_l2_distances.items(), key=lambda x: x[1])
    top_k_similar = []
    for i in range(k):
        top_k_similar.append(squared_l2_distances[i][0])
    return top_k_similar

def build_hash_table(dataset,hash_functions_list):
    # dataset: dictionary of vetors {"...":[...], "...":[...], ...}
    hash_tables = []
    dataset_keys = list(dataset.keys())

    for i in range(len(hash_functions_list)): # number of hash tables
        hash_table = {}
        for key in dataset_keys:
            bin_vector = binary_vector(dataset[key], hash_functions_list[i])
            bin_vector_string = list_to_string(bin_vector)
            if bin_vector_string in hash_table:
                hash_table[bin_vector_string].append(key)
            else: #if new binary vector string in the hash table
                hash_table[bin_vector_string] = [key]
        hash_tables.append(hash_table)        

    return hash_tables

def query_lsh_hash_table(query, dataset, hash_tables, hash_functions_list, k=1):   
    # query: dictionary with 1 vector {"...":[...]}
    # hash_tables: list of disctionary with hash tables
    # hash_functions_list: list of dictionary of hash functions [{"...":[...], "...":[...], ...},...]
    # k: number of top similar vectors to return

    candidate_set = []
    
    query_key = list(query.keys())[0]

    for i in range(len(hash_tables)): # number of hash tables
        hash_table = hash_tables[i]
        query_bin_vector = binary_vector(query[query_key], hash_functions_list[i])
        query_bin_vector_str = list_to_string(query_bin_vector)
        if query_bin_vector_str in hash_table:
            for i in hash_table[query_bin_vector_str]:
                candidate_set.append(i)
    
    if len(candidate_set) < k:
        return candidate_set
    else: 
        dataset_candidate = {}
        for key in candidate_set:
            dataset_candidate[key] = dataset[key]
        return linear(query, dataset_candidate, k)

def binary_vector(vec, hash_functions):
    binary_vec = []
    for hf in hash_functions:
        if dot_product(vec,hash_functions[hf]) > 0:
            binary_vec.append(1)
        else:
            binary_vec.append(0)
    return binary_vec

def dot_product(vec1, vec2):
    D = 0
    for i in range(len(vec1)):
        D+=vec1[i]*vec2[i]
    return D

def hamming_distance(vec1, vec2):
    D = 0
    for i in range(len(vec1)):
        if vec1[i] != vec2[i]: 
            D += 1
    return D

def squared_l2_distance(vec1, vec2):
    D = 0
    for i in range(len(vec1)):
        D += (vec1[i] - vec2[i])**2
    return D

def list_to_string(list):
    s = ''
    for i in range(len(list)):
        s+=str(list[i])
    return s

def generate_hash_functions(n,d,nv):
    hash_funcs=[]
    for j in range(n):
        hash_dict = {}
        for i in range(nv):
            v = np.random.choice([-1, 0, 1], size=d, p=[0.4, 0.2, 0.4]).tolist()
            hash_dict["h"+str(j)+"w" + str(i)] = v
        hash_funcs.append(hash_dict)
    return hash_funcs

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
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_neighbors), k)))
    if idcg == 0:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(approx_neighbors[:k]):
        if doc_id in true_neighbors:
            dcg += 1.0 / np.log2(i + 2)
    return dcg / idcg

df = pd.read_csv('data.csv', index_col=0)
#songIndex 
query_dict = {}
database_dict = {}

#selection of query and columns
queryIndex = 1936

#df.loc[index,col])

#features selected:
columns = ['acousticness','danceability','energy','instrumentalness','liveness','speechiness','valence']

for i in range(len(df)):
    if i!=queryIndex:
        database_dict[str(i)]=df.loc[i,columns].values.tolist() #form database_dict
    else:
        query_dict[str(i)]=df.loc[i,columns].values.tolist() #form query_dict

#parameters
k = 10 #number of nearest neighbours to search for
vec_per_hash_func=10
num_hash_tables=2

hash_dim=len(columns)

np.random.seed(99)
hash_funcs = generate_hash_functions(num_hash_tables, hash_dim, vec_per_hash_func)
start = time.time()
hash_tables = build_hash_table(database_dict, hash_funcs)
end=time.time()
time_buildHashTables = end - start


start = time.time()
knnResult = linear(query_dict, database_dict, k)
end = time.time()
time_knnQuery = end - start

start = time.time()
lshResult = query_lsh_hash_table(query_dict, database_dict, hash_tables, hash_funcs, k)
end = time.time()
time_lshQuery = end - start

print("KNN search runtime:", time_knnQuery, "seconds")
print("KNN search result:",str(knnResult))
'''for i in range(len(knnResult)):
    anySong = df.loc[int(knnResult[i])]
    anySongName = getMusicName(anySong)
    print(knnResult[i], anySongName)'''
print("")
print("LSH hash table runtime (build hash tables):", time_buildHashTables, "seconds")
print("LSH hash table runtime (query):", time_lshQuery, "seconds")
print("LSH hash table result:",str(lshResult))
p = precision(knnResult,lshResult,k)
r = recall(knnResult,lshResult,k)
print(p)
print(r)

#COMPARE function
def KNNvsLSH(query_dict, database_dict, hash_tables, hash_funcs, hash_dim, vec_per_hash_func, num_hash_tables, k):
    start = time.time()
    knnResult = linear(query_dict, database_dict, k)
    end = time.time()
    time_knnQuery = end - start

    start = time.time()
    lshResult = query_lsh_hash_table(query_dict, database_dict, hash_tables, hash_funcs, k)
    end = time.time()
    time_lshQuery = end - start

    p = precision(knnResult, lshResult, k)
    r = recall(knnResult, lshResult, k)
    mrr = mean_reciprocal_rank(knnResult, lshResult)
    ap = average_precision(knnResult, lshResult)
    nd = ndcg(knnResult, lshResult, k)

    return time_knnQuery, time_lshQuery, p, r, mrr, ap, nd


# EXPERIMENT: LSH vs KNN — Runtime & Accuracy under Different Parameters

# Parameter ranges
k_values = [1, 5, 10, 20, 50, 70,100,150]
hash_table_values = [1, 2, 3, 4, 5]
vec_per_hash_values = [1,2,3,4,5, 10, 15, 20, 25]

knn_times_k, lsh_times_k, build_times_k = [], [], []
prec_k, rec_k, mrr_k, map_k, ndcg_k = [], [], [], [], []

knn_times_ht, lsh_times_ht, build_times_ht = [], [], []
prec_ht, rec_ht, mrr_ht, map_ht, ndcg_ht = [], [], [], [], []

knn_times_vec, lsh_times_vec, build_times_vec = [], [], []
prec_vec, rec_vec, mrr_vec, map_vec, ndcg_vec = [], [], [], [], []


# Vary K
np.random.seed(42)
hash_funcs = generate_hash_functions(num_hash_tables, hash_dim, vec_per_hash_func)

# Measure build time
start = time.time()
hash_tables = build_hash_table(database_dict, hash_funcs)
build_time_fixed = time.time() - start
num_runs = 1000  # number of times to repeat and average

for k in k_values:
    avg_t_knn = avg_t_lsh = avg_p = avg_r = avg_mrr = avg_ap = avg_nd = 0.0

    for _ in range(num_runs):
        t_knn, t_lsh, p, r, mrr, ap, nd = KNNvsLSH(
            query_dict, database_dict, hash_tables, hash_funcs,
            hash_dim, vec_per_hash_func, num_hash_tables, k
        )
        avg_t_knn += t_knn
        avg_t_lsh += t_lsh
        avg_p += p
        avg_r += r
        avg_mrr += mrr
        avg_ap += ap
        avg_nd += nd

    # store averaged results
    knn_times_k.append(avg_t_knn / num_runs)
    lsh_times_k.append(avg_t_lsh / num_runs)
    build_times_k.append(build_time_fixed)
    prec_k.append(avg_p / num_runs)
    rec_k.append(avg_r / num_runs)
    mrr_k.append(avg_mrr / num_runs)
    map_k.append(avg_ap / num_runs)
    ndcg_k.append(avg_nd / num_runs)
print(lsh_times_k)
# Vary number of hash tables 
for n_tables in hash_table_values:
    np.random.seed(42)
    hash_funcs = generate_hash_functions(n_tables, hash_dim, vec_per_hash_func)

    # Measure build time
    start = time.time()
    hash_tables = build_hash_table(database_dict, hash_funcs)
    build_time = time.time() - start

    t_knn, t_lsh, p, r, mrr, ap, nd = KNNvsLSH(
        query_dict, database_dict, hash_tables, hash_funcs,
        hash_dim, vec_per_hash_func, n_tables, k=10
    )
    knn_times_ht.append(t_knn)
    lsh_times_ht.append(t_lsh)
    build_times_ht.append(build_time)
    prec_ht.append(p)
    rec_ht.append(r)
    mrr_ht.append(mrr)
    map_ht.append(ap)
    ndcg_ht.append(nd)


# Vary vectors per hash function 
for v in vec_per_hash_values:
    np.random.seed(42)
    hash_funcs = generate_hash_functions(num_hash_tables, hash_dim, v)

    # Measure build time
    start = time.time()
    hash_tables = build_hash_table(database_dict, hash_funcs)
    build_time = time.time() - start

    t_knn, t_lsh, p, r, mrr, ap, nd = KNNvsLSH(
        query_dict, database_dict, hash_tables, hash_funcs,
        hash_dim, v, num_hash_tables, k=10
    )
    knn_times_vec.append(t_knn)
    lsh_times_vec.append(t_lsh)
    build_times_vec.append(build_time)
    prec_vec.append(p)
    rec_vec.append(r)
    mrr_vec.append(mrr)
    map_vec.append(ap)
    ndcg_vec.append(nd)


# Helper to print a table cleanly 
def print_results_table(title, param_name, param_values, knn_times, lsh_times, build_times, prec, rec, mrr, mapv, ndcgv):
    times_faster = np.array(knn_times) / np.array(lsh_times)
    if build_times==None:
        df_results = pd.DataFrame({
            param_name: param_values,
            'Precision': np.round(prec, 3),
            'Recall': np.round(rec, 3),
            'MRR': np.round(mrr, 3),
            'MAP': np.round(mapv, 3),
            'nDCG': np.round(ndcgv, 3),
            'Linear Time (s)': np.round(knn_times, 4),
            'LSH Time (s)': np.round(lsh_times, 4),
            '× Faster (Linear/LSH)': np.round(times_faster, 3)
        })
    else:
        df_results = pd.DataFrame({
            param_name: param_values,
            'Precision': np.round(prec, 3),
            'Recall': np.round(rec, 3),
            'MRR': np.round(mrr, 3),
            'MAP': np.round(mapv, 3),
            'nDCG': np.round(ndcgv, 3),
            'Linear Time (s)': np.round(knn_times, 4),
            'LSH Time (s)': np.round(lsh_times, 4),
            'Build Time (s)': np.round(build_times, 4),
            '× Faster (Linear/LSH)': np.round(times_faster, 3)
        })

    print("\n" + "=" * 100)
    print(f"{title}")
    print("=" * 100)
    print(df_results.to_string(index=False))
    print(f"\nAverages — Precision: {np.mean(prec):.3f}, Recall: {np.mean(rec):.3f}, "
          f"MRR: {np.mean(mrr):.3f}, MAP: {np.mean(mapv):.3f}, nDCG: {np.mean(ndcgv):.3f}")
    print("=" * 100 + "\n")

# Generate 3 results tables 
print_results_table(
    title="(1) LSH vs Linear — Varying K | Build time = "+str(np.round(build_time_fixed, 4))+" s",
    param_name="K",
    param_values=k_values,
    knn_times=knn_times_k,
    lsh_times=lsh_times_k,
    build_times=None,
    prec=prec_k,
    rec=rec_k,
    mrr=mrr_k,
    mapv=map_k,
    ndcgv=ndcg_k
)

print_results_table(
    title="(2) LSH vs Linear — Varying Number of Hash Tables",
    param_name="# Hash Tables",
    param_values=hash_table_values,
    knn_times=knn_times_ht,
    lsh_times=lsh_times_ht,
    build_times=build_times_ht,
    prec=prec_ht,
    rec=rec_ht,
    mrr=mrr_ht,
    mapv=map_ht,
    ndcgv=ndcg_ht
)

print_results_table(
    title="(3) LSH vs Linear — Varying Vectors per Hash Function",
    param_name="Vectors per Hash Function",
    param_values=vec_per_hash_values,
    knn_times=knn_times_vec,
    lsh_times=lsh_times_vec,
    build_times=build_times_vec,
    prec=prec_vec,
    rec=rec_vec,
    mrr=mrr_vec,
    mapv=map_vec,
    ndcgv=ndcg_vec
)

# PLOTTING — Combined Layout (2 rows × 3 columns)
fig, axs = plt.subplots(2, 3, figsize=(17, 8))
fig.suptitle("LSH vs Linear: Runtime and Ranking Metrics Comparison", fontsize=18)

# Runtime (KNN, LSH Query, Build Time) 
# --Runtime vs K
axs[0, 0].plot(k_values, knn_times_k, 'o-', color='red', label='Linear Query')
axs[0, 0].plot(k_values, lsh_times_k, 's-', color='blue', label='LSH Query')
axs[0, 0].plot(k_values, build_times_k, '^-', color='green', label='LSH Build')
axs[0, 0].set_title("Runtime vs K")
axs[0, 0].set_xlabel("K")
axs[0, 0].set_ylabel("Time (seconds)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# --Runtime vs # Hash Tables
axs[0, 1].plot(hash_table_values, knn_times_ht, 'o-', color='red', label='Linear Query')
axs[0, 1].plot(hash_table_values, lsh_times_ht, 's-', color='blue', label='LSH Query')
axs[0, 1].plot(hash_table_values, build_times_ht, '^-', color='green', label='LSH Build')
axs[0, 1].set_title("Runtime vs Number of Hash Tables")
axs[0, 1].set_xlabel("Number of Hash Tables")
axs[0, 1].set_ylabel("Time (seconds)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# --Runtime vs Vectors per Hash Function
axs[0, 2].plot(vec_per_hash_values, knn_times_vec, 'o-', color='red', label='Linear Query')
axs[0, 2].plot(vec_per_hash_values, lsh_times_vec, 's-', color='blue', label='LSH Query')
axs[0, 2].plot(vec_per_hash_values, build_times_vec, '^-', color='green', label='LSH Build')
axs[0, 2].set_title("Runtime vs Vectors per Hash Function")
axs[0, 2].set_xlabel("Vectors per Hash Function")
axs[0, 2].set_ylabel("Time (seconds)")
axs[0, 2].legend()
axs[0, 2].grid(True)

# Row 2: Ranking Metrics
# --Precision, Recall, MRR, MAP, nDCG vs K
axs[1, 0].plot(k_values, prec_k, 'o-', label='Precision', color='green')
axs[1, 0].plot(k_values, rec_k, 's-.', label='Recall', color='orange')
axs[1, 0].plot(k_values, mrr_k, '^:', label='MRR', color='purple')
axs[1, 0].plot(k_values, map_k, 'x:', label='MAP', color='blue')
axs[1, 0].plot(k_values, ndcg_k, 'd:', label='nDCG', color='red')
axs[1, 0].set_title("Ranking Metrics vs K")
axs[1, 0].set_xlabel("K")
axs[1, 0].set_ylabel("Score")
axs[1, 0].set_ylim(0, 1.05)
axs[1, 0].legend()
axs[1, 0].grid(True)

# --Ranking Metrics vs # Hash Tables
axs[1, 1].plot(hash_table_values, prec_ht, 'o-', label='Precision', color='green')
axs[1, 1].plot(hash_table_values, rec_ht, 's-.', label='Recall', color='orange')
axs[1, 1].plot(hash_table_values, mrr_ht, '^:', label='MRR', color='purple')
axs[1, 1].plot(hash_table_values, map_ht, 'x:', label='MAP', color='blue')
axs[1, 1].plot(hash_table_values, ndcg_ht, 'd:', label='nDCG', color='red')
axs[1, 1].set_title("Ranking Metrics vs Number of Hash Tables")
axs[1, 1].set_xlabel("Number of Hash Tables")
axs[1, 1].set_ylabel("Score")
axs[1, 1].set_ylim(0, 1.05)
axs[1, 1].legend()
axs[1, 1].grid(True)

# --Ranking Metrics vs Vectors per Hash Function
axs[1, 2].plot(vec_per_hash_values, prec_vec, 'o-', label='Precision', color='green')
axs[1, 2].plot(vec_per_hash_values, rec_vec, 's-.', label='Recall', color='orange')
axs[1, 2].plot(vec_per_hash_values, mrr_vec, '^:', label='MRR', color='purple')
axs[1, 2].plot(vec_per_hash_values, map_vec, 'x:', label='MAP', color='blue')
axs[1, 2].plot(vec_per_hash_values, ndcg_vec, 'd:', label='nDCG', color='red')
axs[1, 2].set_title("Ranking Metrics vs Vectors per Hash Function")
axs[1, 2].set_xlabel("Vectors per Hash Function")
axs[1, 2].set_ylabel("Score")
axs[1, 2].set_ylim(0, 1.05)
axs[1, 2].legend()
axs[1, 2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
