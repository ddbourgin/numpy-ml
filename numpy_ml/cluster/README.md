# Clustering Models
The `kmeans.py` module implements:

1. [Hard kmeans clustering](https://user-images.githubusercontent.com/1905599/119421132-de04f700-bcb2-11eb-98cd-4337d0b9496d.png) with fixed assignment of data points to only one cluster at a time.
2. [Soft kmeans clustering](https://user-images.githubusercontent.com/1905599/119421211-0bea3b80-bcb3-11eb-9e71-a337da8db24d.png) with probabilistic assignment of data points. Each data point has a membership degree in each cluster. The highest probabe cluster could then be assigned as the cluser index of the data. Alternatively, the probability distribution can be used for any other purpose as it captures our uncertaintity of the clustering routine.

