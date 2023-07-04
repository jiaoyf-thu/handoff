from handoff import*
import time


'''
Here we convert SPH fragments into DEM rigid clusters
'''


def test():
  # input_path = "./data/Particles0015.csv" # "./data/Particles0010.csv" # "./data/test-low.csv" # dist = 1.5 # 0.5 # 0.001 #
  input_path = "D:/jiaoyf/Seafile/SPH/sphsol-para/Data/pd_cn/rock_30m_off_5m/500k/Particles0020.csv"
  # input_path = "D:/jiaoyf/Seafile/SPH/sphsol-para/Data/pd_cn/rock_30m_off_5m_weak/500k/Particles0020.csv"
  # input_path = "D:/jiaoyf/Seafile/SPH/sphsol-para/Data/pd_cn/rock_30m_off_2.5m/Particles0020.csv"  dist = 0.3   mass = 62.37
  # input_path = "D:/jiaoyf/Seafile/SPH/sphsol-para/Data/pd_cn/rock_22m_off_0m/Particles0020.csv"   dist = 0.225   mass = 26.31
  # input_path = "D:/jiaoyf/Seafile/SPH/sphsol-para/Data/pd_cn/rock_30m_off_5m_rp/Particles0020.csv"
  dist = 0.3
  mass = 62.37
  vmax = 0.5
  omg = np.array([0., 0., 2.*np.pi/3600.])

  # load sph particles and generate clusters
  sph_clusters = load_sph_particles(input_path, dist, vmax, mass, omg)

  # handoff from sph fragments into DEM rigid clusters
  dem_clusters = handoff(sph_clusters, dist=dist, rmin=dist)

  # export dem input data
  export_dem_transformed(dem_clusters)


if __name__ == "__main__":
  print(time.asctime(time.localtime(time.time())))
  test()
  print(time.asctime(time.localtime(time.time())))
