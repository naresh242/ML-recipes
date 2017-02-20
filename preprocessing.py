from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer


#rescaling
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]
scale=MinMaxScaler(feature_range=(0,1))
rescaledX=scale.fit_transform(X)
set_printoptions(precision=3)
print(rescaledX[0:5,:])

#standardize Data
# Standardize data (0 mean, 1 stdev)
X=array[:,0:8]
scale=StandardScaler().fit(X)
rescaledX=scale.transform(X)
set_printoptions(precision=3)
print(rescaledX[0:5,:])

# Normalize Data
X=array[:,0:8]
scale=Normalizer().fit(X)
rescaledX=scale.transform(X)
set_printoptions(precision=3)
print(rescaledX[0:5,:])

#Binarize Data
X = array[:,0:8]
Y = array[:,8]
binarizer = Binarizer(threshold=0.0)
binaryX = binarizer.transform(X)
set_printoptions(precision=3)
print(binaryX[0:5,:])


