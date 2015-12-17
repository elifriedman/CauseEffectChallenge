import data_io
import numpy as np
from sklearn.neighbors.kde import KernelDensity

try:
  data
except:
  data = data_io.read_train_pairs()
  vals = data.values
  targ = data_io.read_train_target()
  info = data_io.read_train_info()

def normalizedata():
  for i in range(len(vals)):
    m = np.array([vals[i][0],vals[i][1]])
    mn = m.min(1).reshape(2,1)
    m = m-mn
    mx = m.max(1).reshape(2,1)
    m = m/mx
    vals[i][0] = m[0,:]
    vals[i][1] = m[1,:]
  return vals

def getgridcoords(N):
  s = 0.5/N
  i = np.linspace(s,1-s,N)
  x,y = np.meshgrid(i,i)
  x=x.reshape((N**2))
  y=y.reshape((N**2))
  return np.array([x,y])

N=10
def createfeatmat(N):
  grid = getgridcoords(N).T
  featmat = np.zeros((len(vals),N**2))
  for i in range(len(vals)):
    m = np.array([vals[i][0],vals[i][1]]).T
    k = KernelDensity(bandwidth=0.5/(N-1),kernel='gaussian')
    k.fit(m)
    featmat[i,:] = k.score_samples(grid)
  return featmat


if __name__ == "__main__":
  pass