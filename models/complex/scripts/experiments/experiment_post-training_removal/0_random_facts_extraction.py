import random
from collections import defaultdict

from scipy import cluster
from kelpie.dataset import FB15K, Dataset
import numpy
d = Dataset(FB15K)

entity_2_degree = defaultdict(lambda: 0)
for (head, _, tail) in d.train_triples:
    entity_2_degree[head] += 1
    entity_2_degree[tail] += 1

head_2_degree = {}
for (head, relation, tail) in d.test_triples:
    head_2_degree[head] = entity_2_degree[head]

# extract clusters with k-means
values = [float(v) for v in head_2_degree.values()]
centroids, distortion = cluster.vq.kmeans(numpy.array(values), k_or_guess=3, iter=1000, thresh=1e-05)

small_degree_cluster, mid_degree_cluster, high_degree_cluster = [], [], []
limit1 = int(centroids[0] + centroids[1])/2
limit2 = int(centroids[1] + centroids[2])/2

print(limit1)
print(limit2)
for (h, r, t) in d.test_triples:
    if entity_2_degree[h] < limit1:
        small_degree_cluster.append((h, r, t))
    elif entity_2_degree[h] < limit2:
        mid_degree_cluster.append((h, r, t))
    else:
        high_degree_cluster.append((h, r, t))

print()
# extract 33 test facts for each cluster
small_degree_facts = random.choices(small_degree_cluster, k=33)
mid_degree_facts = random.choices(mid_degree_cluster, k=33)
high_degree_facts = random.choices(high_degree_cluster, k=33)

for x in small_degree_facts:
    print(x)
print()
for x in mid_degree_facts:
    print(x)
print()
for x in high_degree_facts:
    print(x)
print()
