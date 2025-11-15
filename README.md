# Data-Mining-Technical-Review-Report

This project conducts a technical review of similarity search algorithms, focusing on both Exact and Approximate search techniques. The Exact Search Algorithm, Brute Force, is used as a benchmark to obtain the true nearest neighbours. The Exact and Approximate Search Algorithms explored were KD-Tree, Ball Tree, Locality-Sensitive Hashing (LSH), Approximate Nearest Neighbours Oh Yeah (ANNOY), and Hierarchical Navigable Small-World (HNSW). Each algorithm was tested on numerical, word and image datasets to assess its accuracy, runtime performance, complexity, and real-world applicability. While Exact Search provided perfect accuracy both theoretically and empirically, it was computationally expensive for large datasets. 

Approximate search methods, particularly HNSW and ANNOY, achieved accuracy with substantially lower query times, making them more practical for large-scale applications. Tree-based algorithms such as KD-Tree and Ball Tree excel in low-dimensional data, while LSH had varying performances, depending on the nature of the dataset.  The study concludes that algorithm selection should be guided by dataset dimensionality and application requirements to achieve an optimal balance between accuracy and efficiency.

Worked on this project with the following members:
1. Lim Sin Pei
2. Kieran Voo E Kai
3. Clarabelle Chua Jia Yi 
