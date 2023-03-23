import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,euclidean
from scipy.spatial import KDTree
from scipy.linalg import eigh,inv,norm
from scipy.spatial.transform import Rotation as R
import open3d as o3d


# # contact detection

# data = pd.read_csv("./data/edge_particles_pos.txt", sep="\s+", header=None, names=["m","i","x","y","z","r"])
# cluster_id = pd.read_csv("./data/cluster_id.txt", sep="\s+", header=None, names=["cid"])
# cluster_id = np.array(cluster_id)
# rmax = max(data["r"])
# rmin = min(data["r"])
# total_num = len(data.index)
# print("  Edge particle radius in [%f, %f]" % (rmin, rmax))
# # add all particles into kdtree
# xyz = np.array(data[["x","y","z"]])
# kd_tree = KDTree(xyz)

# pairs = kd_tree.query_pairs(r=rmax*2.0,output_type="ndarray")
# print(pairs.shape)
# pairs = pairs[cluster_id[pairs[:,0],0] != cluster_id[pairs[:,1],0],:]
# for line in range(np.shape(pairs)[0]):
#   i,j = pairs[line,0], pairs[line,1]
#   ikey,jkey = cluster_id[i], cluster_id[j]
#   dist = euclidean(xyz[i,:], xyz[j,:])
#   ir = data.loc[i,"r"]
#   jr = data.loc[j,"r"]
#   if ir+jr > dist:
#     print(i,j,ir,jr,ir+jr,dist)


# mass check



print("done")
