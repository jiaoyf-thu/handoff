import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
from alphashape import alphashape
from scipy.spatial.distance import cdist


'''
Here we convert SPH fragments into DEM rigid clusters
'''


class sph_cluster():
  def __init__(self, hash, particles):
    self.hash = hash            # fragment hash
    self.particles = particles  # particles dataframe
    self.mass = 0.0             # total mass
    self.pos = np.zeros(3)      # center of mass
    self.vel = np.zeros(3)      # linear velocity
    self.omg = np.zeros(3)      # angular velocity
    self.J = np.zeros((3,3))    # rotational inertia
  
  def calc(self):
    self.mass = np.sum(self.particles.MASS)
    self.pos = np.array([np.sum(self.particles["X"] * self.particles["MASS"]),\
                        np.sum(self.particles["Y"] * self.particles["MASS"]),\
                        np.sum(self.particles["Z"] * self.particles["MASS"])]) / self.mass
    self.pos = np.array([np.sum(self.particles["VX"] * self.particles["MASS"]),\
                        np.sum(self.particles["VY"] * self.particles["MASS"]),\
                        np.sum(self.particles["VZ"] * self.particles["MASS"])]) / self.mass
    Lox = (self.particles["Y"] * self.particles["VZ"] - self.particles["Z"] * self.particles["VY"]) * self.particles["MASS"]
    Loy = (self.particles["Z"] * self.particles["VX"] - self.particles["X"] * self.particles["VZ"]) * self.particles["MASS"]
    Loz = (self.particles["X"] * self.particles["VY"] - self.particles["Y"] * self.particles["VX"]) * self.particles["MASS"]
    # angular momentum to coordinate origin
    Lo = np.array([np.sum(Lox), np.sum(Loy), np.sum(Loz)])
    # angular momentum to center of mass, Lo = Lc + m*cross(r_oc,v_c)
    Lc = (Lo - self.mass * np.cross(self.pos, self.vel)).transpose()
    # rotational inertia tensor
    self.J = self.calc_inertia_tensor(self.particles, self.pos)
    # calc from Lc = J * omega
    self.omg = np.matmul(LA.inv(self.J), Lc)
  
  def calc_inertia_tensor(self, particles, center):
    pp = pd.DataFrame(columns=["X","Y","Z","MASS"])
    pp.X = particles.X - center[0]
    pp.Y = particles.Y - center[1]
    pp.Z = particles.Z - center[2]
    pp.MASS = particles.MASS
    I_xx = ((np.square(pp.Y) + np.square(pp.Z)) * pp.MASS).sum()
    I_yy = ((np.square(pp.X) + np.square(pp.Z)) * pp.MASS).sum()
    I_zz = ((np.square(pp.X) + np.square(pp.Y)) * pp.MASS).sum()
    I_xy = (-pp.X * pp.Y * pp.MASS).sum()
    I_xz = (-pp.X * pp.Z * pp.MASS).sum()
    I_yz = (-pp.Y * pp.Z * pp.MASS).sum()
    I = np.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])
    return I


def load_sph_particles(path):
  print("  Load sph particles")
  sph_particles = pd.read_csv(path)
  if "MASS" not in sph_particles.columns: sph_particles["MASS"] = 1.0
  isolated_particles = sph_particles[sph_particles.FRAG == -1]
  clustered_particles = sph_particles[sph_particles.FRAG >= 0]

  print("  Record clustered fragments")
  sph_clusters = {}
  for i in sorted(clustered_particles["FRAG"].unique()):
    sph_clusters[i] = sph_cluster(i, clustered_particles[clustered_particles.FRAG == i])
    sph_clusters[i].calc()

  print("  Record isolated particles")
  hash = -1
  for i in isolated_particles.index:
    sph_clusters[hash] = sph_cluster(hash, isolated_particles.loc[i])
    hash -= 1

  return sph_clusters


class dem_cluster():
  def __init__(self, hash, edge_particles):
    self.hash = hash                  # cluster hash
    self.particles = edge_particles   # outer layer particles dataframe
    self.mass = 0.0                   # total mass
    self.pos = np.zeros(3)            # center of mass
    self.vel = np.zeros(3)            # linear velocity
    self.omg = np.zeros(3)            # angular velocity
    self.J = np.zeros((3,3))          # rotational inertia (suppose particle mass = 1)
  
  def add_particle(self, particle):
    self.particles = pd.concat([self.particles, particle])

  # def check(self):
    # ckeck whether satisfy conservation law


def handoff(sph_clusters, dist):
  print("  Start handoff")
  min_size = 50
  alpha = 1.0 / (1.5 * dist)
  dem_clusters = {}

  print("  Generate dem clusters")
  for key in sph_clusters:
    if key < 0 or len(sph_clusters[key].particles.index) < min_size: continue

    # find sph cluster shape and edge particles
    alpha_shape_indices = alphashape(sph_clusters[key].particles[["X","Y","Z"]], alpha)
    edge_particles = sph_clusters[key].particles.iloc[alpha_shape_indices].copy()
    # for edge particles, set radius equal to dist
    edge_particles["R"] = 0.5 * dist

    inner_indices = np.delete(np.array(range(len(sph_clusters[key].particles.index))), alpha_shape_indices)
    inner_particles = sph_clusters[key].particles.iloc[inner_indices].copy()
    inner_particles["R"] = 0.5 * dist
    inner_particles["DIST_EDGE"] = 1.0e9
    dem_clusters[key] = dem_cluster(key, edge_particles)

    # generate inner dem particles
    while len(inner_particles.index) > 1:
      # calculate edt and find the max inner sphere
      center, radius, inner_particles = edt(inner_particles, edge_particles)
      radius -= 0.5 * dist # edge particles radius

      # quit loop if the biggest sphere includes only 1 particle
      dist_ = cdist(center[["X","Y","Z"]], inner_particles[["X","Y","Z"]], "euclidean")
      sphere_indices = np.argwhere(dist_ < radius)[:,-1]
      if len(sphere_indices) <= 1: break

      # for all particles inside the sphere, record as a big dem particle
      sphere_particles = inner_particles.iloc[sphere_indices]
      dem_particle = center.copy()
      dem_particle["R"] = radius
      dem_particle["MASS"] = np.sum(sphere_particles["MASS"])
      dem_particle["VX"] = np.sum(sphere_particles["VX"] * sphere_particles["MASS"]) / dem_particle.MASS
      dem_particle["VY"] = np.sum(sphere_particles["VY"] * sphere_particles["MASS"]) / dem_particle.MASS
      dem_particle["VZ"] = np.sum(sphere_particles["VZ"] * sphere_particles["MASS"]) / dem_particle.MASS
      dem_clusters[key].add_particle(dem_particle)

      # find sphere edge and add these particles into new edge
      alpha_shape_indices = alphashape(sphere_particles[["X","Y","Z"]], alpha)
      edge_particles = sphere_particles.iloc[alpha_shape_indices]
      
      # remove sphere particles from inner particles
      inner_particles = inner_particles.drop(index=sphere_particles.index)

    dem_clusters[key].add_particle(inner_particles)
    print("  SPH to DEM: %d -> %d" % (len(sph_clusters[key].particles.index), len(dem_clusters[key].particles.index)))

  dem_clusters[1].particles.to_csv("./data/dem-cluster1.csv", index=False, header=True)
  print("  Done!")


def edt(inner_particles, edge_particles):
  # memory limit set to 16 G
  ulimit = 16.0 * pow(1024.0, 3.0) / 8.0
  max_inner_num = int(ulimit / len(edge_particles.index))

  if max_inner_num > len(inner_particles.index):
    # euclidean distance map to edge of each inner particle
    dist_ = cdist(inner_particles[["X","Y","Z"]], edge_particles[["X","Y","Z"]], "euclidean")
    # add known dist_edge to dist
    dist_ = np.hstack((dist_, np.array(inner_particles["DIST_EDGE"])[:,np.newaxis]))
    dist_ = np.min(dist_, axis=1)
    inner_particles["DIST_EDGE"] = dist_
    # particle with the max euclidean distance
    center = inner_particles.iloc[[np.argmax(dist_)]] #.to_frame()
  else:
    batch_num = int(np.ceil(len(inner_particles.index) / max_inner_num))
    indices_list = np.array(range(len(inner_particles.index)))
    for i in range(batch_num):
      batch_particles_indices = indices_list[np.where((indices_list+i)%batch_num==0)]
      batch_particles = inner_particles.iloc[batch_particles_indices]
      dist_ = cdist(batch_particles[["X","Y","Z"]], edge_particles[["X","Y","Z"]], "euclidean")
      dist_ = np.hstack((dist_, np.array(batch_particles["DIST_EDGE"])[:,np.newaxis]))
      dist_ = np.min(dist_, axis=1)
      inner_particles.iloc[batch_particles_indices, inner_particles.columns.get_loc("DIST_EDGE")] = dist_
    center = inner_particles.iloc[[inner_particles["DIST_EDGE"].argmax()]]

  radius = np.max(dist_)

  return center, radius, inner_particles
  