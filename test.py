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


# # mass velocity distribution
# path =  "./data/Particles0010.csv" #"./data/test-low.csv" #
# particles = pd.read_csv(path)
# largest = particles[particles.FRAG == 1]
# print(np.mean(largest["VX"]),np.mean(largest["VY"]),np.mean(largest["VZ"]))
# particles["VX"] -= np.mean(largest["VX"])
# particles["VY"] -= np.mean(largest["VY"])
# particles["VZ"] -= np.mean(largest["VZ"])
# particles["vmod"] = np.sqrt(particles["VX"]*particles["VX"] + particles["VY"]*particles["VY"] + particles["VZ"]*particles["VZ"])
# # particles.to_csv("./data/Particles0010_.csv",index=False)

# # cumulative m-v distribution
# print(min(particles["vmod"]),max(particles["vmod"]))
# vlist = np.exp(np.linspace(np.log(1000),np.log(0.001),num=100))
# mlist = np.zeros_like(vlist)

# for i in range(len(vlist)):
#   pp = particles[(particles["vmod"]>vlist[i])]
#   mlist[i] = float(len(pp.index))

# mlist /= float(len(particles.index))

# plt.figure()
# plt.scatter(vlist,mlist,marker=".")
# # plt.xlim(0.01,100)
# # plt.ylim(0.01,1)
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Velocity (m/s)")
# plt.ylabel("Fraction")

# # v direction distribution
# unit_n = np.array([0.916515,0.4,0.0])
# particles["angle"] = np.arccos((particles["VX"]*unit_n[0] + particles["VY"]*unit_n[1] + particles["VZ"]*unit_n[2]) / particles["vmod"]) /np.pi*180.0
# vlist=np.array([0.01,0.1,1.0,10.0])

# # fig,ax = plt.subplots(4,1,sharex=True)
# # fig.set_size_inches(6,12)
# plt.figure()
# colo = plt.cm.viridis(np.linspace(0.2,1.0,4))
# for i in range(len(vlist)):
#   pp = particles[particles["vmod"]>vlist[i]]
#   plt.hist(pp["angle"],bins=36,range=(0,180),color=colo[i],label="v > "+str(vlist[i])+" m/s")

# plt.ylabel("Particles number")
# plt.xlabel("Ejected velocity direction (deg)")
# # plt.yscale("log")
# plt.xlim(0,180)
# plt.legend()
# plt.tight_layout()
# plt.show()

# print("done")


# def hcp(n):
#   k, j, i = [v.flatten() for v in np.meshgrid([range(n[2])],[range(n[1])],[range(n[0])], indexing='ij')]
#   df = pd.DataFrame({
#     'x': 2 * i + (j + k) % 2,
#     'y': np.sqrt(3) * (j + 1/3 * (k % 2)),
#     'z': 2 * np.sqrt(6) / 3 * k,})
#   return df

# # test hcp
# center = np.array([0.,0.,0.])
# box = np.array([30.,15.,30.])
# r = 0.1
# nsize = [int(np.ceil(box[0]/(r*2.))),int(np.ceil(box[1]/(r*np.sqrt(3)))),int(np.ceil(box[2]/(r*2.*np.sqrt(6)/3.)))]
# df = hcp(nsize)*r
# center_ = np.array([(np.max(df["x"])+np.min(df["x"]))/2.0,(np.max(df["y"])+np.min(df["y"]))/2.0,(np.max(df["z"])+np.min(df["z"]))/2.0])
# df["x"] += center[0]-center_[0]
# df["y"] += center[1]-center_[1]
# df["z"] += center[2]-center_[2]

# print(df)
# print("packing percent: ",4./3.*np.pi*pow(r,3)*len(df.index)/(box[0]*box[1]*box[2]))
# df.to_csv("./data/hcp.csv",index=False)


# # test sph_interpolate for sci-imapct1
# print("Current Time =", time.ctime())

# sph_particles = pd.read_csv("D:/jiaoyf/Seafile/SPH/sphsol-para/Data/handoff/sci-impact1/Particles0010.csv")
# sph_particles.drop(sph_particles.head(8).index, inplace=True)
# dem_particles = pd.read_csv("D:/jiaoyf/Seafile/SPH/sphsol-para/Data/handoff/sci-impact1/1100X.csv",\
#     skipinitialspace=True, skiprows=1, names=["X","Y","Z","X0","Y0","Z0","R","U","V","W","CN","W1","W2","W3"])
# dem_particles = dem_particles[["X","Y","Z","R"]]
# # dem_particles = dem_particles.sample(frac=0.01,random_state=1)

# hsml = 0.24
# kappa = 2
# vlimit2 = pow(0.38,2)

# dem_particles = sph_interpolation1(sph_particles, dem_particles, hsml, kappa, vlimit2, kernel)
# dem_particles.to_csv("./data/test_interpolation.csv",index=False)

# print("Current Time =", time.ctime())

# dem_particles = pd.read_csv("./data/test_interpolation.csv")
# dem_particles_ = pd.read_csv("D:/jiaoyf/Seafile/SPH/sphsol-para/Data/handoff/sci-impact1/input_points.txt",\
#     sep="\s+",names=["M","I","X","Y","Z","U","V","W","W1","W2","W3","R"])
# print(dem_particles_.head())
# dem_particles.loc[dem_particles_.index, "R"] = dem_particles_["R"].values
# dem_particles["M"] = 4/3*np.pi*2400*dem_particles["R"]*dem_particles["R"]*dem_particles["R"]
# dem_particles["I"] = 2/5*dem_particles["M"]*dem_particles["R"]*dem_particles["R"]
# dem_particles["W1"] = 0.0
# dem_particles["W2"] = 0.0
# dem_particles["W3"] = 0.0
# dem_particles["U"] = dem_particles["VX"]
# dem_particles["V"] = dem_particles["VY"]
# dem_particles["W"] = dem_particles["VZ"]
# dem_particles = dem_particles[["M","I","X","Y","Z","U","V","W","W1","W2","W3","R"]]
# dem_particles.to_csv("./data/test_interpolation_.csv",index=False,sep=" ")


# # test sph_interpolate for sci-imapct2
# print("Current Time =", time.ctime())

# sph_particles = pd.read_csv("D:/jiaoyf/Seafile/SPH/sphsol-para/Data/handoff/sci-impact2/Particles0010.csv")
# sph_particles.drop(sph_particles.head(8).index, inplace=True)
# sph_particles.loc[sph_particles["PART"]==6, "PART"] = 1
# dem_particles = pd.read_csv("D:/jiaoyf/Seafile/SPH/sphsol-para/Data/handoff/sci-impact2/1100X-stage1.csv",\
#     skipinitialspace=True, skiprows=1, names=["X","Y","Z","X0","Y0","Z0","R","U","V","W","CN","W1","W2","W3"])
# dem_particles = dem_particles[["X","Y","Z","R"]]
# dem_particles_ = pd.read_csv("D:/jiaoyf/Seafile/SPH/sphsol-para/Data/handoff/sci-impact2/input_points.txt",\
#     sep="\s+",names=["M","I","X","Y","Z","U","V","W","W1","W2","W3","R"])
# dem_particles["R"] = dem_particles_["R"].values
# # dem_particles = dem_particles.sample(frac=0.1,random_state=1)

# hsml = 0.24
# kappa = 2
# vlimit2 = pow(0.38,2)

# dem_particles = pre_sph_interpolation2(sph_particles, dem_particles, hsml, kappa, vlimit2, kernel)
# dem_particles["M"] = 4/3*np.pi*2400*dem_particles["R"]*dem_particles["R"]*dem_particles["R"]
# dem_particles["I"] = 2/5*dem_particles["M"]*dem_particles["R"]*dem_particles["R"]
# dem_particles["W1"] = 0.0
# dem_particles["W2"] = 0.0
# dem_particles["W3"] = 0.0
# dem_particles["U"] = dem_particles["VX"]
# dem_particles["V"] = dem_particles["VY"]
# dem_particles["W"] = dem_particles["VZ"]
# dem_particles = dem_particles[["M","I","X","Y","Z","U","V","W","W1","W2","W3","R","PART"]]
# dem_particles.to_csv("./data/sci_impact2_preprocess.csv",index=False,sep=",")
# print("Current Time =", time.ctime())


# # convert to dem assembly
# dem_particles = pd.read_csv("./data/sci_impact2_preprocess.csv")
# # pre_dem_cluster(dem_particles, rmax=0.12, path="./data/sci_impact2_preprocess_.csv")
# pre_dem_cluster_transform(dem_particles, rmax=0.12, path1="./data/sci_impact2/sci_impact2_assembly0.csv", path2="./data/sci_impact2/sci_impact2_points.csv")


# reset dem particles bed only
# dem_particles = pd.read_csv("D:/jiaoyf/Seafile/SPH/sphsol-para/Data/handoff/sci-impact2/1004Assembly", \
#     skipinitialspace=True, skiprows=5, names=["X","Y","Z","U","V","W","W1","W2","W3","Q1","Q2","Q3","Q4"])
# dem_particles_ = pd.read_csv("D:/jiaoyf/Seafile/SPH/sphsol-para/Data/handoff/sci-impact2/input_points.txt",\
#     sep="\s+", skiprows=26783, names=["M","I","Xi","Yi","Zi","R"])
# dem_particles["M"] = dem_particles_["M"].values
# dem_particles["I"] = dem_particles_["I"].values
# dem_particles["Xi"] = 0.0
# dem_particles["Yi"] = 0.0
# dem_particles["Zi"] = 0.0
# dem_particles["R"] = dem_particles_["R"].values
# print(dem_particles.head())

sph_particles = pd.read_csv("D:/jiaoyf/Seafile/SPH/sphsol-para/Data/handoff/sci-impact2/Particles0005.csv")
sph_particles.drop(sph_particles.head(8).index, inplace=True)
sph_particles.loc[sph_particles["PART"]==6, "PART"] = 1
sph_particles["MASS"] = 20.8
sph_particles["R"] = 0.1
boulders = {}
for part_id in [2,3,4,5]:
  boulders[part_id] = sph_cluster(part_id, sph_particles[sph_particles["PART"]==part_id])
  boulders[part_id].calc()
  print(boulders[part_id].vel, boulders[part_id].omg, boulders[part_id].pi, boulders[part_id].mass)

# hsml = 0.24
# kappa = 2
# vlimit2 = pow(1.0,2)

# bed_particles = pre_sph_interpolation2(sph_particles, dem_particles, hsml, kappa, vlimit2, kernel)
# print(bed_particles.head())
# print(bed_particles.shape)


# dem_particles = pd.read_csv("D:/jiaoyf/Seafile/2023Research/sph-dem/handoff/data/sci_impact2/sci_impact2_points_boulders.csv",\
#     sep=" ", names=["M","I","Xi","Yi","Zi","R"])
# density = 4800
# dem_particles["I"] = 2/5*dem_particles["R"]*dem_particles["R"]*density*4/3*np.pi*dem_particles["R"]*dem_particles["R"]*dem_particles["R"]
# dem_particles.loc[:525,"M"] = 11148.8
# dem_particles.loc[525:1060,"M"] = 11148.8
# dem_particles.loc[1060:5152,"M"] = 168812.8
# dem_particles.loc[5152:,"M"] = 1354579.2
# dem_particles.to_csv("./data/sci_impact2/sci_impact2_points_boulders_.csv",index=False,sep=" ",header=False)
