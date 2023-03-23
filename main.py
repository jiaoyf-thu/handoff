from handoff import*
import time


'''
Here we convert SPH fragments into DEM rigid clusters
'''


def test():
  input_path = "./data/test-low.csv" # "./data/Particles0015.csv" # "./data/Particles0010.csv" # "./data/test-low.csv" # 
  dist = 0.001 # 1.5 # 0.5 # 0.001 # 
  vmax = 0.1
  mass = 250.0

  # load sph particles and generate clusters
  sph_clusters = load_sph_particles(input_path, dist, vmax, mass)

  # handoff from sph fragments into DEM rigid clusters
  dem_clusters = handoff(sph_clusters, dist=dist, rmin=dist)

  # export dem input data
  export_dem_transformed(dem_clusters)


if __name__ == "__main__":
  print(time.asctime(time.localtime(time.time())))
  test()
  print(time.asctime(time.localtime(time.time())))
