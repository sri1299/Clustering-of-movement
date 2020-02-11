# Clustering of movement
Dataset: https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F

The dataset contains GPS logs of various users (latitude, longitude, altitude and timestamp). This is converted to cartesian. 
Then each trajectory is represented by a feature vector of average gradient, distance, average altitude, etc. K-Means (centroid-based Euclidean) clustering algorithm is applied with k=5.
# Clusters Desciption
   ## Cluster 0
   ### Noise data. Values are abnormally high.
   ## Cluster 1
   ### Land movement. This could be further clustered into various land vehicles car/bus/train
   ![Alt text](car-bus-train.png?raw=true")
