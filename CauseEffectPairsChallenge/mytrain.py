import data_io
import features as f
import numpy as np
from sklearn.mixture import GMM

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

from sklearn.linear_model import LinearRegression, Perceptron


linreg = LinearRegression(normalize=True)