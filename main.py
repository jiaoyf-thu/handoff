from handoff import*
import time


'''
Here we convert SPH fragments into DEM rigid clusters
'''


def test():
  input_path = "./data/test-low.csv"
  dist = 0.001 #0.25
  vmax = 1.0

  # load sph particles and generate clusters
  sph_clusters = load_sph_particles(input_path, vmax)

  # handoff from sph fragments into DEM rigid clusters
  handoff(sph_clusters, dist=dist, rmin=dist)


if __name__ == "__main__":
  print(time.asctime(time.localtime(time.time())))
  test()
  print(time.asctime(time.localtime(time.time())))
