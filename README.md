# Clustering of movement
Dataset: https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F

The dataset contains GPS logs of various users (latitude, longitude, altitude and timestamp). This is converted to cartesian. 
Then each trajectory is represented by a feature vector of average gradient, distance, average altitude, etc. K-Means (centroid-based Euclidean) clustering algorithm is applied with k=5.

[code for clustering](Clustering_MSGPS.ipynb)
![clustering code image](Clustering.png?raw=true)
# Clusters Desciption
   ## Cluster 0
   ### Noise data. Values are abnormally high.
   ## Cluster 1
   ### Land movement. This could be further clustered into various land vehicles car/bus/train
   ![car/bus/train trajectory](car-bus-train.png?raw=true)
   ## Cluster 2
   ### Foot Movement. Due to the Low speed and small distance.
   ![walking](walking.png?raw=true)
   ## Cluster 3
   ### Air based movement. This has been inferred from the high altitude, speed and distance travelled.
   ![flying vehicles trajectory](flying.png?raw=true)
   ## Cluster 4
   ### Movement in Hilly region, hence the high altitude and moderate speed.
   ![hilly region](hilly.png?raw=true)
