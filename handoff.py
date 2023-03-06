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


def load_sph_particles(path, vmax):
  print("1 Load sph particles")
  sph_particles = pd.read_csv(path)
  if "MASS" not in sph_particles.columns: sph_particles["MASS"] = 1.0
  sph_particles["VMOD2"] = sph_particles["VX"]*sph_particles["VX"] + sph_particles["VY"]*sph_particles["VY"] + sph_particles["VZ"]*sph_particles["VZ"]
  isolated_particles = sph_particles[(sph_particles.FRAG == -1) & (sph_particles.VMOD2 < vmax*vmax)]
  clustered_particles = sph_particles[sph_particles.FRAG >= 0]

  print("  Record clustered fragments")
  sph_clusters = {}
  for i in sorted(clustered_particles["FRAG"].unique()):
    sph_clusters[i] = sph_cluster(i, clustered_particles[clustered_particles.FRAG == i])
    sph_clusters[i].calc()

  print("  Record isolated particles")
  hash = -1
  for i in isolated_particles.index:
    sph_clusters[hash] = sph_cluster(hash, isolated_particles.loc[[i]])
    hash -= 1

  print("  Input %d SPH particles" % len(sph_particles.index))

  return sph_clusters


class dem_cluster():
  def __init__(self, hash, sph_cluster_, edge_particles):
    self.hash = hash              # cluster hash
    self.edge_particles = edge_particles                                # edge particles for contact only
    self.grav_particles = pd.DataFrame(columns=edge_particles.columns)  # particles for gravity only
    self.mass = sph_cluster_.mass # total mass
    self.pos  = sph_cluster_.pos  # center of mass
    self.vel  = sph_cluster_.vel  # linear velocity
    self.omg  = sph_cluster_.omg  # angular velocity
    self.J    = sph_cluster_.J    # rotational inertia (suppose particle mass = 1)
  
  def add_particle(self, particle):
    self.grav_particles = pd.concat([self.grav_particles, particle])

  def check(self):
    coef = self.mass / sum(self.grav_particles.MASS)
    self.grav_particles.MASS *= coef


def handoff(sph_clusters, dist, rmin):
  print("2 Start handoff")
  min_size = 50
  report_size = 1000
  alpha = 1.0 / (1.0 * dist)
  dem_clusters = {}

  print("  Generate dem clusters")
  for key in sph_clusters:
    if key < 0 or len(sph_clusters[key].particles.index) < min_size:
      sph_clusters[key].particles["R"] = 0.5 * dist
      # directly add edge particles and gravitational particles
      dem_clusters[key] = dem_cluster(key, sph_clusters[key], sph_clusters[key].particles)
      dem_clusters[key].add_particle(sph_clusters[key].particles)
      continue

    # find sph cluster shape and edge particles
    sph_clusters[key].particles["R"] = 0.5 * dist
    alpha_shape_indices = alphashape(sph_clusters[key].particles[["X","Y","Z"]], alpha)
    edge_particles = sph_clusters[key].particles.iloc[alpha_shape_indices].copy()
    dem_clusters[key] = dem_cluster(key, sph_clusters[key], edge_particles)

    inner_particles = sph_clusters[key].particles.copy()
    inner_particles["DIST_EDGE"] = np.inf

    # generate inner dem particles
    while len(inner_particles.index) > 1:
      # calculate edt and find the max inner sphere
      center, radius, inner_particles = edt(inner_particles, edge_particles)
      radius += 0.5*dist
      if radius < rmin: break

      # quit loop if the biggest sphere includes only 1 particle
      dist_ = cdist(center[["X","Y","Z"]], inner_particles[["X","Y","Z"]], "euclidean")
      sphere_indices = np.argwhere(dist_ <= radius)[:,-1]
      sphere_edge_indices = np.argwhere((dist_ > radius) & (dist_ <= radius + dist))[:,-1]
      if len(sphere_indices) <= 1 or len(sphere_edge_indices) <= 1: break

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
      edge_particles = inner_particles.iloc[sphere_edge_indices]
      
      # remove sphere particles from inner particles
      inner_particles = inner_particles.drop(index=sphere_particles.index)

    # dem_clusters[key].add_particle(inner_particles)
    if len(sph_clusters[key].particles.index) > report_size:
      print("  SPH to DEM: %d -> %d" % (len(sph_clusters[key].particles.index), len(dem_clusters[key].grav_particles.index)))

  for key in dem_clusters: dem_clusters[key].check()

  dem_grav_pnum = [len(dem_clusters[key].grav_particles.index) for key in dem_clusters]
  dem_edge_pnum = [len(dem_clusters[key].edge_particles.index) for key in dem_clusters]
  print("  Export %d DEM grav_particles" % sum(dem_grav_pnum))
  print("  Export %d DEM edge_particles" % sum(dem_edge_pnum))

  # test
  dem_clusters[1].grav_particles.to_csv("./data/grav_particles.csv", index=False, header=True)
  dem_clusters[1].edge_particles.to_csv("./data/edge_particles.csv", index=False, header=True)

  print("3 Done!")


def edt(inner_particles, edge_particles):
  # memory limit set to 10 G
  ulimit = 10.0 * pow(1024.0, 3.0) / 8.0
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
  