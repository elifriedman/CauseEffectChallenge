import data_io
import features as f
import numpy as np
from sklearn.mixture import GMM

import data_io
import matplotlib.pyplot as plt

try:
  data
except:
  data = data_io.read_train_pairs()
  targ = data_io.read_train_target()
  info = data_io.read_train_info()

def main():
  pass
#  useEMMeans()
#  setCounter()

def plot(i):
  plt.plot(data.A[i],data.B[i],'.b')
#  plt.plot(data.B[i],data.A[i],'.')
  f = plt.gca()
  f.axes.margins(0.1,0.1)
#  f.axes.get_xaxis().set_visible(False)
#  f.axes.get_yaxis().set_visible(False)
#  plt.savefig('imgs/im'+str(i)+'_'+str(targ.Target[i])+'.png',bbox_inches='tight',pad_inches=0)
#  plt.close()

def setCounter():
  total = 0
  nonZero = 0
  for i in range(len(data)):
    dataA = [round(d/5) for d in data.A[i]]
    dataB = [round(d/5) for d in data.B[i]]

    #if len(set(int(data.A[i]/5)))<len(set(int(data.B[i]/5))):
    if len(set(dataA))<len(set(dataB)):
     guess = 1
    else:
      guess = -1
    real = targ.Target[i]
    if real != 0:
      total += (real == guess)
      nonZero +=1
      print(real,len(set(data.A[i])),len(set(data.B[i])))
      plt.figure()
      plot(i)
  print(total,nonZero)

def data2im(i):
  minx = min(data.A[i])
  maxx = max(data.A[i])
  miny = min(data.B[i])
  maxy = max(data.B[i])
#  A =

def useEMMeans():
  data = data_io.read_train_pairs()
  output = data_io.read_train_target()
  y = np.array(output)
  n_components=5
  covariance_type='full' #alternatives 'diag','spherical'
  num_datas = len(data)
  #means = np.zeros((num_datas,n_components,2))
  #for i in range(num_datas):
  #  X = np.array([data.A[i],data.B[i]]).T
  #  g = GMM(n_components=n_components)
  #  g.fit(X)
  #  means[i,:,:] = g.means_

  means = np.load('gmm_means.npy')
  X = means[:,:,1]
  y = y[:,0]
  from sklearn.linear_model import LinearRegression, Perceptron
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.neighbors import KNeighborsClassifier

  k = 20
  n = range(0,len(X)/2+k)
  npp = range(len(X)/2+k,len(X))

  def dostuff(csfr,X,y,n,npp):
    csfr.fit(X[n,:],y[n])
    yhat = csfr.predict(X[npp,:])
    print 1.0*sum(yhat==y[npp])/len(yhat)

  linreg = LinearRegression(normalize=True)
  dostuff(linreg,X,y,n,npp)

  p = Perceptron()
  dostuff(p,X,y,n,npp)

  dt = DecisionTreeClassifier()
  dostuff(dt,X,y,n,npp)

  knn = KNeighborsClassifier(n_neighbors=2)
  dostuff(knn,X,y,n,npp)

  r = np.random.randint(-1,3,len(y[npp]))
  r[r==2] = 0
  print 1.0*sum(r==y[npp])/len(r)



if __name__ == "__main__":
  main()