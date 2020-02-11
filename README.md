# Clustering of movement
Dataset: https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F
The dataset contains GPS logs of various users (latitude, longitude, altitude and timestamp). This is converted to cartesian. 
Then each trajectory is represented by a feature vector of average gradient, distance, average altitude, etc. K-Means (centroid-based Euclidean) clustering algorithm is applied with k=5.
# Clusters Desciption
## Cluster 0: 
    This cluster should be junk because of very less number of trajectories and abnormal values 
    of speed and distance.

## Cluster 1:
    Land Movement in Car/Train/Bus. This could be further clustered into various types of land vehicles.
    ![Moving in car/bus](/cluster_1.png)
## Cluster 2: 
    Foot Movement. Low speed and small distance.

## Cluster 3:
    Movement in air. High speed and altitude.

## Cluster 4:
    Movement in Hilly region, high altitude.
