import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,euclidean
from scipy.spatial import KDTree,Delaunay
from scipy.linalg import eigh,inv,norm
from scipy.interpolate import RBFInterpolator
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import time


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
    self.quat = np.array([0.0,0.0,0.0,1.0])     # quaternions: in scalar-last (x, y, z, w) format
    self.pi = np.zeros(3)       # principle interia
  
  def calc(self):
    if len(self.particles.index) == 1:
      self.mass = self.particles.iloc[0,self.particles.columns.get_loc("MASS")]
      self.pos = np.array([self.particles.iloc[0,self.particles.columns.get_loc("X")], \
                          self.particles.iloc[0,self.particles.columns.get_loc("Y")],\
                          self.particles.iloc[0,self.particles.columns.get_loc("Z")]])
      self.vel = np.array([self.particles.iloc[0,self.particles.columns.get_loc("VX")], \
                          self.particles.iloc[0,self.particles.columns.get_loc("VY")],\
                          self.particles.iloc[0,self.particles.columns.get_loc("VZ")]])
      # if self.hash==-1 : print(self.hash, self.pos, self.vel)
      # self.pos = np.array([self.particles.X, self.particles.Y, self.particles.Z])
      # self.vel = np.array([self.particles.VX, self.particles.VY, self.particles.VZ])
      self.pi += 0.4 * self.mass * pow(self.particles.iloc[0,self.particles.columns.get_loc("R")], 2.0)
    else:
      self.mass = np.sum(self.particles.MASS)
      self.pos = np.array([np.sum(self.particles["X"] * self.particles["MASS"]),\
                          np.sum(self.particles["Y"] * self.particles["MASS"]),\
                          np.sum(self.particles["Z"] * self.particles["MASS"])]) / self.mass
      self.vel = np.array([np.sum(self.particles["VX"] * self.particles["MASS"]),\
                          np.sum(self.particles["VY"] * self.particles["MASS"]),\
                          np.sum(self.particles["VZ"] * self.particles["MASS"])]) / self.mass
      # print(self.hash, self.pos, self.vel)
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
      self.omg = np.matmul(inv(self.J), Lc)
      # calc quaternion
      self.calc_quaternion()
  
  def calc_inertia_tensor(self, particles, center):
    pp = particles[["X","Y","Z","MASS","R"]].copy()
    pp.X -= center[0]
    pp.Y -= center[1]
    pp.Z -= center[2]
    Ic = (0.4 * pp.MASS * np.square(pp.R)).sum()
    I_xx = ((np.square(pp.Y) + np.square(pp.Z)) * pp.MASS).sum() + Ic
    I_yy = ((np.square(pp.X) + np.square(pp.Z)) * pp.MASS).sum() + Ic
    I_zz = ((np.square(pp.X) + np.square(pp.Y)) * pp.MASS).sum() + Ic
    I_xy = (-pp.X * pp.Y * pp.MASS).sum()
    I_xz = (-pp.X * pp.Z * pp.MASS).sum()
    I_yz = (-pp.Y * pp.Z * pp.MASS).sum()
    I = np.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])
    return I

  def calc_quaternion(self):
    w, v = eigh(self.J)
    self.pi = w
    r = R.from_matrix(v)
    self.quat = r.as_quat()
    # print(self.hash, v, r.as_matrix())

  def transform(self):
    # transfer all particles position into inertia coordinate
    r = R.from_quat(self.quat)
    A = (r.as_matrix()).transpose()
    for index in self.particles.index:
      pos0 = np.array([self.particles.loc[index,"X"], self.particles.loc[index,"Y"], self.particles.loc[index,"Z"]])
      pos0 -= self.pos
      pos1 = np.matmul(A,pos0)
      self.particles.loc[index,"X"] = pos1[0]
      self.particles.loc[index,"Y"] = pos1[1]
      self.particles.loc[index,"Z"] = pos1[2]


def load_sph_particles(path, dist, vmax=1.0, mass=1.0, omg=np.zeros(3), move2center=True):
  print("1 Load sph particles")
  sph_particles = pd.read_csv(path)
  if "MASS" not in sph_particles.columns: sph_particles["MASS"] = mass
  sph_particles["R"] = 0.5 * dist

  # add an initial rotation: omg cross pos
  sph_particles["VX"] += omg[1] * sph_particles["Z"] - omg[2] * sph_particles["Y"]
  sph_particles["VY"] += omg[2] * sph_particles["X"] - omg[0] * sph_particles["Z"]
  sph_particles["VZ"] += omg[0] * sph_particles["Y"] - omg[1] * sph_particles["X"]

  # move to the largest fragment
  largest = sph_particles[sph_particles.FRAG == 1]
  if len(largest) > 0 and move2center:
    largest_pos = np.array([np.sum(largest["X"] * largest["MASS"]),\
                            np.sum(largest["Y"] * largest["MASS"]),\
                            np.sum(largest["Z"] * largest["MASS"])]) / np.sum(largest["MASS"])
    largest_vel = np.array([np.sum(largest["VX"] * largest["MASS"]),\
                            np.sum(largest["VY"] * largest["MASS"]),\
                            np.sum(largest["VZ"] * largest["MASS"])]) / np.sum(largest["MASS"])
  else:
    largest_pos = np.array([np.sum(sph_particles["X"] * sph_particles["MASS"]),\
                            np.sum(sph_particles["Y"] * sph_particles["MASS"]),\
                            np.sum(sph_particles["Z"] * sph_particles["MASS"])]) / np.sum(sph_particles["MASS"])
    largest_vel = np.array([np.sum(sph_particles["VX"] * sph_particles["MASS"]),\
                            np.sum(sph_particles["VY"] * sph_particles["MASS"]),\
                            np.sum(sph_particles["VZ"] * sph_particles["MASS"])]) / np.sum(sph_particles["MASS"])
  print("Largest fragment pos: ", largest_pos)
  print("Largest fragment vel: ", largest_vel)
  sph_particles["X"] -= largest_pos[0]
  sph_particles["Y"] -= largest_pos[1]
  sph_particles["Z"] -= largest_pos[2]
  sph_particles["VX"] -= largest_vel[0]
  sph_particles["VY"] -= largest_vel[1]
  sph_particles["VZ"] -= largest_vel[2]
  # delete too fast particles
  # sph_particles["V2COM"] = (sph_particles["VX"]*sph_particles["X"] + sph_particles["VY"]*sph_particles["Y"] + sph_particles["VZ"]*sph_particles["Z"]) /\
  #                          np.sqrt(sph_particles["X"]*sph_particles["X"] + sph_particles["Y"]*sph_particles["Y"] + sph_particles["Z"]*sph_particles["Z"])
  # sph_particles = sph_particles[~((sph_particles.FRAG == -1) & (sph_particles.V2COM >= vmax))]
  sph_particles["VMOD2"] = sph_particles["VX"]*sph_particles["VX"] + sph_particles["VY"]*sph_particles["VY"] + sph_particles["VZ"]*sph_particles["VZ"]
  sph_particles = sph_particles[~((sph_particles.FRAG == -1) & (sph_particles.VMOD2 >= vmax*vmax))]
  
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
    sph_clusters[hash] = sph_cluster(hash, isolated_particles.loc[[i]])
    sph_clusters[hash].calc()
    hash -= 1

  print("  Input %d SPH particles" % len(sph_particles.index))
  print("  Input %d SPH clusters" % len(sph_clusters))

  return sph_clusters


class dem_cluster():
  def __init__(self, hash, sph_cluster_, mesh=None):
    self.hash = hash              # cluster hash
    # self.edge_particles = pd.DataFrame(columns=sph_cluster_.particles.columns)  # edge particles for contact only
    self.grav_particles = pd.DataFrame(columns=sph_cluster_.particles.columns)  # particles for gravity only
    self.mass = sph_cluster_.mass # total mass
    self.pos  = sph_cluster_.pos  # center of mass
    self.vel  = sph_cluster_.vel  # linear velocity
    self.omg  = sph_cluster_.omg  # angular velocity
    self.J    = sph_cluster_.J    # rotational inertia
    self.quat = sph_cluster_.quat # quaternions: in scalar-last (x, y, z, w) format
    self.pi   = sph_cluster_.pi   # principle interia
    self.mesh = mesh
    # if self.mesh == None:
    #   self.edge_particles = sph_cluster_.particles
    # else:
    #   self.edge_particles = pd.DataFrame(np.asarray(self.mesh.vertices),columns=["X","Y","Z"])
    self.edge_particles = mesh[["X","Y","Z"]]

  def add_particle(self, particle):
    self.grav_particles = pd.concat([self.grav_particles, particle])

  def check(self, dist, mass):
    # modify mass distribution
    coef = self.mass / sum(self.grav_particles.MASS)
    self.grav_particles.MASS *= coef

    self.edge_particles["R"] = 0.5 * dist
    self.edge_particles["MASS"] = mass
    # # move edge particles inside and set them larger
    # if self.mesh != None:
    #   normals = pd.DataFrame(np.asarray(self.mesh.vertex_normals),columns=["NX","NY","NZ"]) # direction outward
    #   # pcd = o3d.geometry.PointCloud()
    #   # pcd.points = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
    #   # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=dist*2.0, max_nn=15), fast_normal_computation=False)
    #   # pcd.normalize_normals()
    #   # pcd.normals = o3d.utility.Vector3dVector(orient_normals(pcd))
    #   # normals = pd.DataFrame(np.asarray(pcd.normals),columns=["NX","NY","NZ"])
    #   self.edge_particles["X"] -= normals["NX"] * 0.25 * dist
    #   self.edge_particles["Y"] -= normals["NY"] * 0.25 * dist
    #   self.edge_particles["Z"] -= normals["NZ"] * 0.25 * dist
    #   self.edge_particles["R"] = 0.75 * dist
    #   self.edge_particles["MASS"] = mass

  def transform(self):
    # transfer all particles position into inertia coordinate
    r = R.from_quat(self.quat)
    A = (r.as_matrix()).transpose()
    for index in self.grav_particles.index:
      pos0 = np.array([self.grav_particles.loc[index,"X"], self.grav_particles.loc[index,"Y"], self.grav_particles.loc[index,"Z"]])
      pos0 -= self.pos
      pos1 = np.matmul(A,pos0)
      self.grav_particles.loc[index,"X"] = pos1[0]
      self.grav_particles.loc[index,"Y"] = pos1[1]
      self.grav_particles.loc[index,"Z"] = pos1[2]
    for index in self.edge_particles.index:
      pos0 = np.array([self.edge_particles.loc[index,"X"], self.edge_particles.loc[index,"Y"], self.edge_particles.loc[index,"Z"]])
      pos0 -= self.pos
      pos1 = np.matmul(A,pos0)
      self.edge_particles.loc[index,"X"] = pos1[0]
      self.edge_particles.loc[index,"Y"] = pos1[1]
      self.edge_particles.loc[index,"Z"] = pos1[2]


# # failed! :(
# def orient_normals(pcd):
#   points = np.asarray(pcd.points)
#   normals = np.asarray(pcd.normals)
#   # breadth first search
#   kd_tree = KDTree(points)
#   node=0
#   visited = []
#   queue = []
#   visited.append(node)
#   queue.append(node)
#   while queue:
#     s = queue.pop(0)
#     dd,ii = kd_tree.query(points[s,:],k=5)
#     for neighbour in ii:
#       if neighbour not in visited:
#         visited.append(neighbour)
#         queue.append(neighbour)
#         normals[neighbour,:] *= (-1.0 if np.dot(normals[neighbour,:],normals[s,:])<0.0 else 1.0)
#   return normals


def check_contact(dem_clusters_):
  # check particles contact
  dem_clusters = dem_clusters_.copy()
  rmax = max([(dem_clusters[key].edge_particles["R"]).max() for key in dem_clusters])
  rmin = min([(dem_clusters[key].edge_particles["R"]).min() for key in dem_clusters])
  total_num = sum([len(dem_clusters[key].edge_particles.index) for key in dem_clusters])
  print("  Edge particle radius in [%f, %f]" % (rmin, rmax))
  # add all particles into kdtree
  xyz = pd.DataFrame(columns=["X","Y","Z"])
  cluster_id = np.zeros(total_num,dtype=int)
  cum_num = 0
  for key in dem_clusters:
    xyz = pd.concat([xyz, dem_clusters[key].edge_particles[["X","Y","Z"]]])
    edge_num = len(dem_clusters[key].edge_particles.index)
    dem_clusters[key].edge_particles["PCDID"] = cum_num + np.array(range(edge_num),dtype=int)
    cluster_id[cum_num+np.array(range(edge_num),dtype=int)] = key
    cum_num += edge_num
  xyz = np.array(xyz)
  kd_tree = KDTree(xyz)

  # np.savetxt("./data/cluster_id.txt",cluster_id)
  pairs = kd_tree.query_pairs(r=rmax*2.0,output_type="ndarray")
  pairs = pairs[cluster_id[pairs[:,0]] != cluster_id[pairs[:,1]],:]
  for line in range(np.shape(pairs)[0]):
    i,j = pairs[line,0], pairs[line,1]
    ikey,jkey = cluster_id[i], cluster_id[j]
    dist = euclidean(xyz[i,:], xyz[j,:])
    ix = (dem_clusters[ikey].edge_particles[dem_clusters[ikey].edge_particles["PCDID"]==i]).index[0]
    jx = (dem_clusters[jkey].edge_particles[dem_clusters[jkey].edge_particles["PCDID"]==j]).index[0]
    ir = dem_clusters[ikey].edge_particles.loc[ix,"R"]
    jr = dem_clusters[jkey].edge_particles.loc[jx,"R"]
    r_sum = ir+jr
    if r_sum >= dist:
      ir *= 0.99 * dist / r_sum # (ir+jr)
      jr *= 0.99 * dist / r_sum # (ir+jr)
      dem_clusters[ikey].edge_particles.loc[ix,"R"] = ir
      dem_clusters[jkey].edge_particles.loc[jx,"R"] = jr

  return dem_clusters


def handoff(sph_clusters, dist, rmin):
  print("2 Start handoff")
  min_size = 10
  report_size = 100
  # alpha = 1.0 / (1.5 * dist)
  dem_clusters = {}

  # record the first point from each cluster
  points_to_test = pd.DataFrame([sph_clusters[key].particles.iloc[0][["X", "Y", "Z"]] for key in sph_clusters], columns=["X", "Y", "Z"])
  points_to_test.set_index(pd.Index([key for key in sph_clusters]), inplace=True)
  points_to_test["is_inside"] = False
  key_inside = []

  print("  Generate dem clusters")
  for key in sph_clusters:
    # ignore inside clusters
    if key in key_inside: continue

    # ignore too fast rotating clusters
    if norm(sph_clusters[key].omg) > 10.0:
      print("  Cluster %d is rotating too fast" % key)
      continue

    sph_clusters[key].particles["R"] = 0.5 * dist
    if key < 0 or len(sph_clusters[key].particles.index) < min_size:
      # directly add edge particles and gravitational particles
      dem_clusters[key] = dem_cluster(key, sph_clusters[key], sph_clusters[key].particles[["X","Y","Z"]])
      # Suppose only 1 grav_particle
      particle = sph_clusters[key].particles.iloc[[0]].copy()
      particle.X = sph_clusters[key].pos[0]
      particle.Y = sph_clusters[key].pos[1]
      particle.Z = sph_clusters[key].pos[2]
      particle.VX = sph_clusters[key].vel[0]
      particle.VY = sph_clusters[key].vel[1]
      particle.VZ = sph_clusters[key].vel[2]
      dem_clusters[key].add_particle(particle)
      continue

    # find sph cluster shape and edge particles
    xyz = (sph_clusters[key].particles[["X","Y","Z"]]).copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,1.5*dist)
    # mesh.compute_vertex_normals(normalized=True)
    # # orient normals outward
    # normals = np.asarray(mesh.vertex_normals)
    # points = np.asarray(mesh.vertices)
    # kd_tree = KDTree(xyz)
    # _,neighbors = kd_tree.query(points,k=10)
    # for ii in range(points.shape[0]):
    #   center = np.mean(xyz[neighbors[ii,:],:],axis=0)
    #   normals[ii,:] *= 1.0 if np.dot(normals[ii,:], (points[ii,:]-center)) > 0.0 else -1.0
    # mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    # first layer
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,2.0*dist) #1.5*dist)
    edge_particles = pd.DataFrame(np.asarray(mesh.vertices), columns=["X","Y","Z"])
    edge_particles_ = edge_particles.copy()
    edge_particles["BID"] = 0
    for i in range(len(edge_particles)):
      edge_particles.loc[i,"BID"] = xyz[(xyz["X"]==edge_particles.loc[i,"X"]) & \
        (xyz["Y"]==edge_particles.loc[i,"Y"]) & \
        (xyz["Z"]==edge_particles.loc[i,"Z"])].index[0]

    # remove inside particles or fragments
    scene = o3d.t.geometry.RaycastingScene()
    mesh.remove_unreferenced_vertices()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh)
    points_to_test['is_inside'] = scene.compute_occupancy(o3d.core.Tensor(points_to_test[['X','Y','Z']].values, dtype=o3d.core.Dtype.Float32)) > 0
    points_to_test.loc[key,'is_inside'] = False # ignore itself
    key_inside += points_to_test[points_to_test['is_inside']==True].index.tolist()

    # second layer
    xyz.drop(edge_particles["BID"], inplace=True)
    if len(xyz.index) > 4:
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
      mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,1.5*dist)
      edge2_particles = pd.DataFrame(np.asarray(mesh.vertices), columns=["X","Y","Z"])
      edge_particles = pd.concat([edge_particles, edge2_particles], ignore_index=True)
      dem_clusters[key] = dem_cluster(key, sph_clusters[key], edge_particles)
    else:
      dem_clusters[key] = dem_cluster(key, sph_clusters[key], sph_clusters[key].particles[["X","Y","Z"]])
    
    edge_particles = edge_particles_.copy()

    # edge_particles = pd.DataFrame(np.asarray(mesh.vertices),columns=["X","Y","Z"])
    # dem_clusters[key] = dem_cluster(key, sph_clusters[key], mesh)

    inner_particles = sph_clusters[key].particles.copy()
    inner_particles["DIST_EDGE"] = np.inf

    # generate inner dem particles
    while len(inner_particles.index) > 1:
      # calculate edt and find the max inner sphere
      if len(edge_particles.index)==0: break
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

    # in case only 1 grav_particles added, move it to com
    if len(dem_clusters[key].grav_particles.index) == 1:
      dem_clusters[key].grav_particles.loc[dem_clusters[key].grav_particles.index[0],"X"] = dem_clusters[key].pos[0]
      dem_clusters[key].grav_particles.loc[dem_clusters[key].grav_particles.index[0],"Y"] = dem_clusters[key].pos[1]
      dem_clusters[key].grav_particles.loc[dem_clusters[key].grav_particles.index[0],"Z"] = dem_clusters[key].pos[2]

    # in case no grav_particles added
    if len(dem_clusters[key].grav_particles.index) == 0:
      particle = sph_clusters[key].particles.iloc[[0]].copy()
      particle.X = sph_clusters[key].pos[0]
      particle.Y = sph_clusters[key].pos[1]
      particle.Z = sph_clusters[key].pos[2]
      particle.VX = sph_clusters[key].vel[0]
      particle.VY = sph_clusters[key].vel[1]
      particle.VZ = sph_clusters[key].vel[2]
      dem_clusters[key].add_particle(particle)

    # report each cluster's handoff
    if len(sph_clusters[key].particles.index) >= report_size:
      print("  SPH to DEM: %d -> %d" % (len(sph_clusters[key].particles.index), len(dem_clusters[key].grav_particles.index)))

  print("  Check dem particles")
  for key in dem_clusters: dem_clusters[key].check(dist, sph_clusters[-1].particles.iloc[0,sph_clusters[-1].particles.columns.get_loc("MASS")])

  print("  Check contact particles")
  dem_clusters = check_contact(dem_clusters)

  # test only
  export_dem_inertial(dem_clusters)

  print("  Coordinate transform")
  for key in dem_clusters: dem_clusters[key].transform()

  return dem_clusters


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
  

def export_dem_transformed(dem_clusters):
  # report handoff results
  dem_grav_pnum = [len(dem_clusters[key].grav_particles.index) for key in dem_clusters]
  dem_edge_pnum = [len(dem_clusters[key].edge_particles.index) for key in dem_clusters]
  print("3 Generate DEM input formats")
  print("  Export %d DEM grav_particles" % sum(dem_grav_pnum))
  print("  Export %d DEM edge_particles" % sum(dem_edge_pnum))

  # report edge particles size
  rmax = max([(dem_clusters[key].edge_particles["R"]).max() for key in dem_clusters])
  rmin = min([(dem_clusters[key].edge_particles["R"]).min() for key in dem_clusters])
  print("  Edge particle radius in [%f, %f]" % (rmin, rmax))

  fp1 = open("./data/dem_input/input_assembly.txt", "w")
  fp2 = open("./data/dem_input/input_points.txt", "w")
  fp3 = open("./data/dem_input/input_points_interior.txt", "w")

  fp1.write("%d\n" % (len(dem_clusters)))
  fp3.write("%d\n" % (sum(dem_grav_pnum)))

  for key in dem_clusters:
    dc = dem_clusters[key]
    fp1.write("%d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n" % (len(dc.edge_particles.index),\
            dc.mass, dc.pi[0],dc.pi[1],dc.pi[2], dc.pos[0],dc.pos[1],dc.pos[2], dc.vel[0],dc.vel[1],dc.vel[2],\
            dc.omg[0],dc.omg[1],dc.omg[2], dc.quat[0],dc.quat[1],dc.quat[2],dc.quat[3]))
    for index in dc.edge_particles.index:
      fp2.write("%.10e %.10e %.10e %.10e %.10e %.10e\n" % (dc.mass, 0.4*dc.edge_particles.loc[index,"MASS"]*pow(dc.edge_particles.loc[index,"R"],2.0),\
               dc.edge_particles.loc[index,"X"],dc.edge_particles.loc[index,"Y"],\
               dc.edge_particles.loc[index,"Z"], dc.edge_particles.loc[index,"R"]))
    fp3.write("%d\n" % (len(dc.grav_particles.index)))
    for index in dc.grav_particles.index:
      fp3.write("%.10e %.10e %.10e %.10e\n" % (dc.grav_particles.loc[index,"MASS"], dc.grav_particles.loc[index,"X"],dc.grav_particles.loc[index,"Y"],\
               dc.grav_particles.loc[index,"Z"]))

  fp1.close()
  fp2.close()
  fp3.close()

  print("4 Done!")


def export_dem_inertial(dem_clusters):
  fp4 = open("./data/dem_input/input_points_edge_inertial.txt", "w")
  fp5 = open("./data/dem_input/input_points_grav_inertial.txt", "w")

  fp4.write("cluster,x,y,z,r\n")
  fp5.write("cluster,x,y,z,r\n")

  for key in dem_clusters:
    dc = dem_clusters[key]
    for index in dc.edge_particles.index:
      fp4.write("%d,%.10e,%.10e,%.10e,%.10e\n" % (key, dc.edge_particles.loc[index,"X"],dc.edge_particles.loc[index,"Y"],\
               dc.edge_particles.loc[index,"Z"], dc.edge_particles.loc[index,"R"]))
    for index in dc.grav_particles.index:
      fp5.write("%d,%.10e,%.10e,%.10e,%.10e\n" % (key, dc.grav_particles.loc[index,"X"],dc.grav_particles.loc[index,"Y"],\
               dc.grav_particles.loc[index,"Z"], dc.grav_particles.loc[index,"R"]))

  fp4.close()
  fp5.close()


def export_edt(inner_particles, num):
  inner_particles[["X","Y","Z","DIST_EDGE"]].to_csv("./data/dem_input/edt"+str(num)+".txt",index=False)


def kernel(rij, h):
  h1 = 1. / h
  q = rij * h1
  # get the kernel normalizing factor
  fac = 1.0 / np.pi * h1 * h1 * h1
  tmp2 = 2. - q
  if q > 2.0:
    val = 0.0
  elif q > 1.0:
    val = 0.25 * tmp2 * tmp2 * tmp2
  else:
    val = 1 - 1.5 * q * q * (1 - 0.5 * q)
  return val * fac


def sph_interpolation1(sph_particles, dem_particles, hsml, kappa, vlimit2, kernel):
  sph_particles = sph_particles.copy()
  dem_particles = dem_particles.copy()
  # sph_particles["VMOD2"] = sph_particles["VX"]*sph_particles["VX"] + sph_particles["VY"]*sph_particles["VY"] + sph_particles["VZ"]*sph_particles["VZ"]
  dem_particles["VX"] = 0.0
  dem_particles["VY"] = 0.0
  dem_particles["VZ"] = 0.0

  # kernel interpolation
  print("Interpolating velocity...")
  # core_particles = sph_particles[sph_particles["VMOD2"] < vlimit2].copy()
  core_particles = sph_particles[sph_particles["Z"] < hsml*0.5].copy()
  sph_xyz = core_particles[["X","Y","Z"]].values
  sph_uvw = core_particles[["VX","VY","VZ"]].values
  dem_xyz = dem_particles[["X","Y","Z"]].values
  sph_tree = KDTree(sph_xyz)
  dem_tree = KDTree(dem_xyz)
  results = dem_tree.query_ball_tree(sph_tree, r=hsml*kappa)
  vol = pow(hsml/1.2, 3)
  for i in range(len(results)):
    if i%10000 == 0: print("%d/%d" % (i,len(results)))
    if len(results[i])==0: continue
    dist = cdist(dem_xyz[i,:].reshape(1,-1),sph_xyz[results[i],:])
    w = np.array([kernel(dist_, hsml) for dist_ in dist[0]]).transpose()
    dem_particles.loc[i,["VX","VY","VZ"]] = np.matmul(w, sph_uvw[results[i],:]) * vol

  # add ejceted particles
  print("Adding ejected particles...")
  ejeceted_particles = sph_particles[sph_particles["Z"] >= hsml*0.5].copy()
  ejeceted_particles["R"] = hsml/2.4
  ejeceted_particles.reset_index(drop=True, inplace=True)

  # check contact
  print("Checking contact...")
  rmax = max(dem_particles["R"].max(), ejeceted_particles["R"].max())
  ejetcted_xyz = ejeceted_particles[["X","Y","Z"]].values
  ejetcted_tree = KDTree(ejetcted_xyz)
  results = ejetcted_tree.query_ball_tree(dem_tree, r=rmax*2.0)
  for i in range(len(results)):
    if len(results[i])==0: continue
    ir = ejeceted_particles.loc[i,"R"]
    dist = cdist(ejetcted_xyz[i,:].reshape(1,-1),dem_xyz[results[i],:])[0]
    for j in range(len(results[i])):
      jr = dem_particles.loc[results[i][j],"R"]
      r_sum = ir+jr
      if r_sum >= dist[j]:
        ir *= 0.99 * dist[j] / r_sum
        jr *= 0.99 * dist[j] / r_sum
        ejeceted_particles.loc[i,"R"] = ir
        dem_particles.loc[results[i],"R"] = jr
  results = ejetcted_tree.query_ball_tree(ejetcted_tree, r=rmax*2.0)
  for i in range(len(results)):
    if len(results[i])==0: continue
    ir = ejeceted_particles.loc[i,"R"]
    dist = cdist(ejetcted_xyz[i,:].reshape(1,-1),ejetcted_xyz[results[i],:])[0]
    for j in range(len(results[i])):
      if results[i][j]==i: continue
      jr = ejeceted_particles.loc[results[i][j],"R"]
      r_sum = ir+jr
      if r_sum >= dist[j]:
        ir *= 0.99 * dist[j] / r_sum
        jr *= 0.99 * dist[j] / r_sum
        ejeceted_particles.loc[i,"R"] = ir
        ejeceted_particles.loc[results[i][j],"R"] = jr

  dem_particles = pd.concat([dem_particles, ejeceted_particles[["X","Y","Z","R","VX","VY","VZ"]]], ignore_index=True, axis=0)
  # dem_particles = dem_particles.append(ejeceted_particles[["X","Y","Z","R","VX","VY","VZ"]], ignore_index=True)
  dem_particles.reset_index(drop=True, inplace=True)
  print("rmax = %f, rmin = %f" % (dem_particles["R"].max(), dem_particles["R"].min()))
  
  return dem_particles


def pre_sph_interpolation2(sph_particles, dem_particles, hsml, kappa, vlimit2, kernel):
  sph_particles = sph_particles.copy()
  dem_particles = dem_particles.copy()
  sph_particles["VMOD2"] = sph_particles["VX"]*sph_particles["VX"] + sph_particles["VY"]*sph_particles["VY"] + sph_particles["VZ"]*sph_particles["VZ"]

  # kernel interpolation
  print("Interpolating velocity...")
  core_particles = sph_particles[sph_particles["VMOD2"] < vlimit2].copy()
  core_particles.reset_index(drop=True, inplace=True)
  sph_xyz = core_particles[["X","Y","Z"]].values
  sph_uvw = core_particles[["VX","VY","VZ"]].values
  bed_particles = dem_particles.copy()
  bed_particles.reset_index(drop=True, inplace=True)
  dem_xyz = bed_particles[["X","Y","Z"]].values
  sph_tree = KDTree(sph_xyz)
  dem_tree = KDTree(dem_xyz)
  results = dem_tree.query_ball_tree(sph_tree, r=hsml*kappa)
  vol = pow(hsml/1.2, 3)
  for i in range(len(results)):
    if i%10000 == 0: print("%d/%d" % (i,len(results)))
    if len(results[i])==0:
      bed_particles.drop(i, inplace=True)
      continue
    dist = cdist(dem_xyz[i,:].reshape(1,-1),sph_xyz[results[i],:])
    w = np.array([kernel(dist_, hsml) for dist_ in dist[0]]).transpose()
    bed_particles.loc[i,["U","V","W"]] = np.matmul(w, sph_uvw[results[i],:]) * vol

  # export
  print("Export...")
  path1 = "./data/sci_impact2/sci_impact2_assembly_bed.csv"
  path2 = "./data/sci_impact2/sci_impact2_points_bed.csv"
  fp1 = open(path1, "w")
  fp2 = open(path2, "w")
  for index in bed_particles.index:
    fp1.write("%d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n" % (1,\
            bed_particles.loc[index,"M"], bed_particles.loc[index,"I"], bed_particles.loc[index,"I"], bed_particles.loc[index,"I"], \
            bed_particles.loc[index,"X"], bed_particles.loc[index,"Y"], bed_particles.loc[index,"Z"], bed_particles.loc[index,"U"], \
            bed_particles.loc[index,"V"], bed_particles.loc[index,"W"], 0., 0., 0., 0., 0., 0., 1.))
    fp2.write("%.10e %.10e %.10e %.10e %.10e %.10e\n" % (bed_particles.loc[index,"M"], bed_particles.loc[index,"I"], \
              0., 0., 0., bed_particles.loc[index,"R"]))
  fp1.close()
  fp2.close()
  
  return bed_particles


def sph_interpolation2(sph_particles, dem_particles, hsml, kappa, vlimit2, kernel):
  sph_particles = sph_particles.copy()
  sph_particles["VMOD2"] = sph_particles["VX"]*sph_particles["VX"] + sph_particles["VY"]*sph_particles["VY"] + sph_particles["VZ"]*sph_particles["VZ"]
  dem_particles = dem_particles.copy()
  dem_particles["VX"] = 0.0
  dem_particles["VY"] = 0.0
  dem_particles["VZ"] = 0.0
  dem_particles["PART"] = 0.0
  sph_particles.reset_index(drop=True, inplace=True)
  dem_particles.reset_index(drop=True, inplace=True)

  # kernel interpolation
  print("Interpolating parts...")
  core_particles = sph_particles[sph_particles["VMOD2"] < vlimit2].copy()
  # core_particles = sph_particles[sph_particles["Z"] < hsml*0.5].copy()
  sph_xyz = core_particles[["X","Y","Z"]].values
  sph_part = core_particles[["PART"]].values
  dem_xyz = dem_particles[["X","Y","Z"]].values
  sph_tree = KDTree(sph_xyz)
  dem_tree = KDTree(dem_xyz)
  results = dem_tree.query_ball_tree(sph_tree, r=hsml)
  for i in range(len(results)):
    if i%10000 == 0: print("%d/%d" % (i,len(results)))
    if len(results[i])==0: continue
    dem_particles.loc[i,"PART"] = np.argmax(np.bincount(sph_part[results[i],0]))

  dem_particles["PART"] = np.round(dem_particles["PART"])
  # dem_particles["PART"] = dem_particles["PART"].astype(int)
  dem_particles = dem_particles[dem_particles["PART"] > 0].copy()
  
  return dem_particles


def pre_dem_cluster(dem_particles, rmax, path):
  martix = dem_particles[dem_particles["PART"]==1].copy()
  martix.to_csv(path, index=False, mode="w")
  boulders = {}
  for part_id in [2,3,4,5]:
    boulder_particles = dem_particles[dem_particles["PART"]==part_id].copy()
    boulder_particles["MASS"] = boulder_particles["M"]
    boulder_particles["VX"] = boulder_particles["U"]
    boulder_particles["VY"] = boulder_particles["V"]
    boulder_particles["VZ"] = boulder_particles["W"]
    boulder_particles_ = boulder_particles.copy()
    boulders[part_id] = sph_cluster(part_id, boulder_particles)
    boulders[part_id].calc()

    # first layer
    print("First layer...")
    xyz = np.array(boulder_particles[["X","Y","Z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,2.0*rmax)
    edge_particles = pd.DataFrame(np.asarray(mesh.vertices), columns=["X","Y","Z"])
    edge_particles["BID"] = 0
    for i in range(len(edge_particles)):
      edge_particles.loc[i,"BID"] = boulder_particles_[(boulder_particles_["X"]==edge_particles.loc[i,"X"]) & \
        (boulder_particles_["Y"]==edge_particles.loc[i,"Y"]) & \
        (boulder_particles_["Z"]==edge_particles.loc[i,"Z"])].index[0]

    # second layer
    print("Second layer...")
    boulder_particles.drop(edge_particles["BID"], inplace=True)
    xyz = np.array(boulder_particles[["X","Y","Z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,2.0*rmax)
    edge2_particles = pd.DataFrame(np.asarray(mesh.vertices), columns=["X","Y","Z"])
    edge2_particles["BID"] = 0
    for i in range(len(edge2_particles)):
      edge2_particles.loc[i,"BID"] = boulder_particles_[(boulder_particles_["X"]==edge2_particles.loc[i,"X"]) & \
        (boulder_particles_["Y"]==edge2_particles.loc[i,"Y"]) & \
        (boulder_particles_["Z"]==edge2_particles.loc[i,"Z"])].index[0]
    edge_particles = pd.concat([edge_particles, edge2_particles], ignore_index=True)

    # third layer
    print("Third layer...")
    boulder_particles.drop(edge2_particles["BID"], inplace=True)
    xyz = np.array(boulder_particles[["X","Y","Z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,2.0*rmax)
    edge3_particles = pd.DataFrame(np.asarray(mesh.vertices), columns=["X","Y","Z"])
    edge3_particles["BID"] = 0
    for i in range(len(edge3_particles)):
      edge3_particles.loc[i,"BID"] = boulder_particles_[(boulder_particles_["X"]==edge3_particles.loc[i,"X"]) & \
        (boulder_particles_["Y"]==edge3_particles.loc[i,"Y"]) & \
        (boulder_particles_["Z"]==edge3_particles.loc[i,"Z"])].index[0]
    edge_particles = pd.concat([edge_particles, edge3_particles], ignore_index=True)

    edge_particles["R"] = boulder_particles_.loc[edge_particles["BID"],"R"].values
    edge_particles["M"] = boulder_particles_.loc[edge_particles["BID"],"M"].values
    edge_particles["I"] = boulder_particles_.loc[edge_particles["BID"],"I"].values
    edge_particles["U"] = 0.0
    edge_particles["V"] = 0.0
    edge_particles["W"] = 0.0
    edge_particles["W1"] = 0.0
    edge_particles["W2"] = 0.0
    edge_particles["W3"] = 0.0
    edge_particles["PART"] = part_id
    edge_particles = edge_particles[["M","I","X","Y","Z","U","V","W","W1","W2","W3","R","PART"]]
    boulders[part_id].particles = edge_particles
    # boulders[part_id].transform()
    boulders[part_id].particles.to_csv(path, index=False, header=False, mode="a")


def pre_dem_cluster_transform(dem_particles, rmax, path1, path2):
  martix = dem_particles[dem_particles["PART"]==1].copy()
  boulders = {}
  for part_id in [2,3,4,5]:
    boulder_particles = dem_particles[dem_particles["PART"]==part_id].copy()
    boulder_particles["MASS"] = boulder_particles["M"]
    boulder_particles["VX"] = boulder_particles["U"]
    boulder_particles["VY"] = boulder_particles["V"]
    boulder_particles["VZ"] = boulder_particles["W"]
    boulder_particles_ = boulder_particles.copy()
    boulders[part_id] = sph_cluster(part_id, boulder_particles)
    boulders[part_id].calc()

    # first layer
    print("First layer...")
    xyz = np.array(boulder_particles[["X","Y","Z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,2.0*rmax)
    edge_particles = pd.DataFrame(np.asarray(mesh.vertices), columns=["X","Y","Z"])
    edge_particles["BID"] = 0
    for i in range(len(edge_particles)):
      edge_particles.loc[i,"BID"] = boulder_particles_[(boulder_particles_["X"]==edge_particles.loc[i,"X"]) & \
        (boulder_particles_["Y"]==edge_particles.loc[i,"Y"]) & \
        (boulder_particles_["Z"]==edge_particles.loc[i,"Z"])].index[0]

    # second layer
    print("Second layer...")
    boulder_particles.drop(edge_particles["BID"], inplace=True)
    xyz = np.array(boulder_particles[["X","Y","Z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,2.0*rmax)
    edge2_particles = pd.DataFrame(np.asarray(mesh.vertices), columns=["X","Y","Z"])
    edge2_particles["BID"] = 0
    for i in range(len(edge2_particles)):
      edge2_particles.loc[i,"BID"] = boulder_particles_[(boulder_particles_["X"]==edge2_particles.loc[i,"X"]) & \
        (boulder_particles_["Y"]==edge2_particles.loc[i,"Y"]) & \
        (boulder_particles_["Z"]==edge2_particles.loc[i,"Z"])].index[0]
    edge_particles = pd.concat([edge_particles, edge2_particles], ignore_index=True)

    # third layer
    print("Third layer...")
    boulder_particles.drop(edge2_particles["BID"], inplace=True)
    xyz = np.array(boulder_particles[["X","Y","Z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd,2.0*rmax)
    edge3_particles = pd.DataFrame(np.asarray(mesh.vertices), columns=["X","Y","Z"])
    edge3_particles["BID"] = 0
    for i in range(len(edge3_particles)):
      edge3_particles.loc[i,"BID"] = boulder_particles_[(boulder_particles_["X"]==edge3_particles.loc[i,"X"]) & \
        (boulder_particles_["Y"]==edge3_particles.loc[i,"Y"]) & \
        (boulder_particles_["Z"]==edge3_particles.loc[i,"Z"])].index[0]
    edge_particles = pd.concat([edge_particles, edge3_particles], ignore_index=True)

    edge_particles["R"] = boulder_particles_.loc[edge_particles["BID"],"R"].values
    edge_particles["M"] = boulder_particles_.loc[edge_particles["BID"],"M"].values
    edge_particles["I"] = boulder_particles_.loc[edge_particles["BID"],"I"].values
    edge_particles["U"] = 0.0
    edge_particles["V"] = 0.0
    edge_particles["W"] = 0.0
    edge_particles["W1"] = 0.0
    edge_particles["W2"] = 0.0
    edge_particles["W3"] = 0.0
    edge_particles["PART"] = part_id
    edge_particles = edge_particles[["M","I","X","Y","Z","U","V","W","W1","W2","W3","R","PART"]]
    boulders[part_id].particles = edge_particles
    boulders[part_id].transform()

  # export
  print("Export...")
  fp1 = open(path1, "w")
  fp2 = open(path2, "w")
  fp1.write("%d\n" % (len(boulders)+len(martix.index)))
  for key in boulders:
    dc = boulders[key]
    # fp1.write("%d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n" % (len(dc.particles.index),\
    #         dc.mass, dc.pi[0],dc.pi[1],dc.pi[2], dc.pos[0],dc.pos[1],dc.pos[2], dc.vel[0],dc.vel[1],dc.vel[2],\
    #         dc.omg[0],dc.omg[1],dc.omg[2], dc.quat[0],dc.quat[1],dc.quat[2],dc.quat[3]))
    fp1.write("%d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n" % (len(dc.particles.index),\
            dc.mass, dc.pi[0],dc.pi[1],dc.pi[2], dc.pos[0],dc.pos[1],dc.pos[2], 0., 0., 0., 0., 0., 0., dc.quat[0],dc.quat[1],dc.quat[2],dc.quat[3]))
    for index in dc.particles.index:
      fp2.write("%.10e %.10e %.10e %.10e %.10e %.10e\n" % (dc.mass, 0.4*dc.particles.loc[index,"M"]*pow(dc.particles.loc[index,"R"],2.0),\
               dc.particles.loc[index,"X"],dc.particles.loc[index,"Y"], dc.particles.loc[index,"Z"], dc.particles.loc[index,"R"]))
  for index in martix.index:
    # fp1.write("%d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n" % (1,\
    #         martix.loc[index,"M"], martix.loc[index,"I"], martix.loc[index,"I"], martix.loc[index,"I"], \
    #         martix.loc[index,"X"], martix.loc[index,"Y"], martix.loc[index,"Z"], martix.loc[index,"U"], \
    #         martix.loc[index,"V"], martix.loc[index,"W"], 0., 0., 0., 0., 0., 0., 1.))
    fp1.write("%d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n" % (1,\
            martix.loc[index,"M"], martix.loc[index,"I"], martix.loc[index,"I"], martix.loc[index,"I"], \
            martix.loc[index,"X"], martix.loc[index,"Y"], martix.loc[index,"Z"], 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.))
    fp2.write("%.10e %.10e %.10e %.10e %.10e %.10e\n" % (martix.loc[index,"M"], martix.loc[index,"I"], \
              0., 0., 0., martix.loc[index,"R"]))
  fp1.close()
  fp2.close()
  