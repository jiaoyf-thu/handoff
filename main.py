from handoff import*
import time


'''
Here we convert SPH fragments into DEM rigid clusters
'''


def test():
  input_path = "./data/Particles0010.csv"
  dist = 0.25 #0.001 #0.25

  # load sph particles and generate clusters
  sph_clusters = load_sph_particles(input_path)

  # handoff from sph fragments into DEM rigid clusters
  handoff(sph_clusters, dist)


if __name__ == "__main__":
  print(time.asctime(time.localtime(time.time())))
  test()
  print(time.asctime(time.localtime(time.time())))
