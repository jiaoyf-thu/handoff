import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,euclidean
from scipy.spatial import KDTree
from scipy.linalg import eigh,inv,norm
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from handoff import *

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


# mass velocity distribution
path =  "./data/Particles0010.csv" #"./data/test-low.csv" #
particles = pd.read_csv(path)
largest = particles[particles.FRAG == 1]
print(np.mean(largest["VX"]),np.mean(largest["VY"]),np.mean(largest["VZ"]))
particles["VX"] -= np.mean(largest["VX"])
particles["VY"] -= np.mean(largest["VY"])
particles["VZ"] -= np.mean(largest["VZ"])
particles["vmod"] = np.sqrt(particles["VX"]*particles["VX"] + particles["VY"]*particles["VY"] + particles["VZ"]*particles["VZ"])
# particles.to_csv("./data/Particles0010_.csv",index=False)

# cumulative m-v distribution
print(min(particles["vmod"]),max(particles["vmod"]))
vlist = np.exp(np.linspace(np.log(1000),np.log(0.001),num=100))
mlist = np.zeros_like(vlist)

for i in range(len(vlist)):
  pp = particles[(particles["vmod"]>vlist[i])]
  mlist[i] = float(len(pp.index))

mlist /= float(len(particles.index))

plt.figure()
plt.scatter(vlist,mlist,marker=".")
# plt.xlim(0.01,100)
# plt.ylim(0.01,1)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Fraction")

# v direction distribution
unit_n = np.array([0.916515,0.4,0.0])
particles["angle"] = np.arccos((particles["VX"]*unit_n[0] + particles["VY"]*unit_n[1] + particles["VZ"]*unit_n[2]) / particles["vmod"]) /np.pi*180.0
vlist=np.array([0.01,0.1,1.0,10.0])

# fig,ax = plt.subplots(4,1,sharex=True)
# fig.set_size_inches(6,12)
plt.figure()
colo = plt.cm.viridis(np.linspace(0.2,1.0,4))
for i in range(len(vlist)):
  pp = particles[particles["vmod"]>vlist[i]]
  plt.hist(pp["angle"],bins=36,range=(0,180),color=colo[i],label="v > "+str(vlist[i])+" m/s")

plt.ylabel("Particles number")
plt.xlabel("Ejected velocity direction (deg)")
# plt.yscale("log")
plt.xlim(0,180)
plt.legend()
plt.tight_layout()
plt.show()

print("done")
