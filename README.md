# Clustering of movement
Dataset: https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F

The dataset contains GPS logs of various users (latitude, longitude, altitude and timestamp). This is converted to cartesian. 
Then each trajectory is represented by a feature vector of average gradient, distance, average altitude, etc. K-Means (centroid-based Euclidean) clustering algorithm is applied with k=5.

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from kmeans_pytorch import kmeans,kmeans_predict\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv('Feature_rep.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "X=data[['alt_mean','alt_max','norm_diff_xy','diff_z_mean','dist']].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_norm=X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(5):\n",
    "    X_max=X[:,i].max()\n",
    "    X_min=X[:,i].min()\n",
    "    X_norm[:,i]=X_norm[:,i]-X_min\n",
    "    X_norm[:,i]=X_norm[:,i]/(X_max-X_min)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2.24102055e-01, 2.26380192e-01, 7.14068452e-05, 3.97169173e-01,\n",
       "        6.56364175e-03],\n",
       "       [2.23817881e-01, 2.29238528e-01, 7.24353336e-05, 3.97080517e-01,\n",
       "        1.29277268e-02],\n",
       "       [2.23142251e-01, 2.23492916e-01, 6.32697556e-05, 3.97352424e-01,\n",
       "        1.67036532e-03],\n",
       "       ...,\n",
       "       [2.49693635e-01, 2.49685519e-01, 2.03054910e-05, 3.97297539e-01,\n",
       "        4.37976652e-04],\n",
       "       [2.49752986e-01, 2.49875448e-01, 2.84448983e-05, 3.97181460e-01,\n",
       "        4.80534620e-04],\n",
       "       [2.49545257e-01, 2.49590555e-01, 6.00744622e-05, 3.97422987e-01,\n",
       "        6.59871903e-04]])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_norm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "if torch.cuda.is_available():\n",
    "    device=torch.device('cuda:0')\n",
    "else:\n",
    "    device=torch.device('cpu')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "device(type='cuda', index=0)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "device"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_t=torch.from_numpy(X_norm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "running k-means on cuda:0..\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[running kmeans]: 17it [00:00, 17.65it/s, center_shift=0.000062, iteration=17, tol=0.000100]\n"
     ]
    }
   ],
   "source": [
    "cluster_id, cluster_centers = kmeans(\n",
    "    X=X_t, num_clusters=5, distance='euclidean', device=device\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[2.4089e-01, 2.4628e-01, 2.8363e-03, 3.9726e-01, 4.4800e-01],\n",
       "        [2.2492e-01, 2.2682e-01, 4.6064e-04, 3.9727e-01, 8.8334e-02],\n",
       "        [2.2348e-01, 2.2396e-01, 1.1941e-04, 3.9726e-01, 3.3138e-03],\n",
       "        [3.2594e-01, 3.3276e-01, 6.7686e-04, 3.9812e-01, 1.2256e-02],\n",
       "        [2.6380e-01, 2.6579e-01, 2.6599e-04, 3.9664e-01, 3.4462e-03]])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cluster_centers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "cluster_id=cluster_id.numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.insert(8,\"C_id\",cluster_id,True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "data_grp_cid=data.groupby(['C_id'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([0, 1, 2, 3, 4], dtype=int64),\n",
       " array([   531,   3911, 152138,   1016,   2934], dtype=int64))"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.unique(cluster_id,return_counts=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>alt_mean</th>\n",
       "      <th>alt_max</th>\n",
       "      <th>norm_diff_xy</th>\n",
       "      <th>diff_z_mean</th>\n",
       "      <th>norm_mean</th>\n",
       "      <th>norm_std</th>\n",
       "      <th>dist</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>C_id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>6.378929e+06</td>\n",
       "      <td>6.379156e+06</td>\n",
       "      <td>5384.742316</td>\n",
       "      <td>-0.032038</td>\n",
       "      <td>5384.745046</td>\n",
       "      <td>4066.261440</td>\n",
       "      <td>5.638924e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>6.378256e+06</td>\n",
       "      <td>6.378337e+06</td>\n",
       "      <td>874.546967</td>\n",
       "      <td>-0.016156</td>\n",
       "      <td>874.549629</td>\n",
       "      <td>391.457710</td>\n",
       "      <td>1.111844e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>6.378195e+06</td>\n",
       "      <td>6.378216e+06</td>\n",
       "      <td>226.695494</td>\n",
       "      <td>-0.034232</td>\n",
       "      <td>226.708336</td>\n",
       "      <td>125.070694</td>\n",
       "      <td>4.171014e+04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>6.382511e+06</td>\n",
       "      <td>6.382799e+06</td>\n",
       "      <td>1285.044918</td>\n",
       "      <td>0.820535</td>\n",
       "      <td>1285.223404</td>\n",
       "      <td>197.464631</td>\n",
       "      <td>1.542614e+05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>6.379894e+06</td>\n",
       "      <td>6.379978e+06</td>\n",
       "      <td>504.985566</td>\n",
       "      <td>-0.641078</td>\n",
       "      <td>505.122244</td>\n",
       "      <td>222.130453</td>\n",
       "      <td>4.337685e+04</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          alt_mean       alt_max  norm_diff_xy  diff_z_mean    norm_mean  \\\n",
       "C_id                                                                       \n",
       "0     6.378929e+06  6.379156e+06   5384.742316    -0.032038  5384.745046   \n",
       "1     6.378256e+06  6.378337e+06    874.546967    -0.016156   874.549629   \n",
       "2     6.378195e+06  6.378216e+06    226.695494    -0.034232   226.708336   \n",
       "3     6.382511e+06  6.382799e+06   1285.044918     0.820535  1285.223404   \n",
       "4     6.379894e+06  6.379978e+06    504.985566    -0.641078   505.122244   \n",
       "\n",
       "         norm_std          dist  \n",
       "C_id                             \n",
       "0     4066.261440  5.638924e+06  \n",
       "1      391.457710  1.111844e+06  \n",
       "2      125.070694  4.171014e+04  \n",
       "3      197.464631  1.542614e+05  \n",
       "4      222.130453  4.337685e+04  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_grp_cid.mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Clusters Desciption\n",
    "## Cluster 0: \n",
    "    This cluster should be junk because of very less number of trajectories and abnormal values of speed and distance.\n",
    "## Cluster 1:\n",
    "    Land Movement in Car/Train/Bus. This could be further clustered into various types of land vehicles.\n",
    "## Cluster 2: \n",
    "    Foot Movement. Low speed and small distance.\n",
    "## Cluster 3:\n",
    "    Movement in air. High speed and altitude.\n",
    "## Cluster 4:\n",
    "    Movement in Hilly region, high altitude.\n",
    "    "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

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
