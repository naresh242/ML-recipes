from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

filename='C:/Users/na347632/Desktop/FAI/learning/python_pro_bundle/machine_learning_mastery_with_python/code/chapter_05/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe=read_csv(filename,names=names)
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]

#univariate selection
test=SelectKBest(score_func=chi2,k=4)
fit=test.fit(X,Y)
features=fit.transform(X)
print(fit.scores_)
print(features[0:5,:])

# Recursive Feature Elimination
model=LogisticRegression()
fit=RFE(model,4)
featureselect=fit.fit(X,Y)
print(featureselect.n_features_)
print(featureselect.support_)
print(featureselect.ranking_)

#PCA
pcamo=PCA(n_components=4)
fit=pcamo.fit(X)
print(fit.explained_variance_)
print(fit.components_)

#feature importance using ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)



