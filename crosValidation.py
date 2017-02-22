from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression

filename=''
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe=read_csv(filename,names=names)
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]

#Split into train and test sets
seed=7
testSize=.33
X_train,X_test,Y_train,Y_test=sklearn.model_selection.train_test_split(X,Y,test_size=testSize,random_state=seed)
model=LogisticRegression()
model.fit(X_train,Y_train)
accuracy=model.score(X_test,Y_test)
print(accuracy)

# kfold cross validation
seed=7
nfold=10
kfold=KFold(n_splits=nfold,random_state=seed)
model=LogisticRegression()
result=cross_val_score(model,X,Y,cv=kfold)
print(result)
print(result.mean())
print(result.std())

#leave one out cross-validation
loom=LeaveOneOut()
model=LogisticRegression()
result=cross_val_score(model,X,Y,cv=loom)
print(result.mean())
print(result.std())

#Repeated Random test-train split
nsplit=10
seed=7
test_size=0.33
rfold=ShuffleSplit(n_splits=nsplit,test_size=test_size,random_state=seed)
result=cross_val_score(model,X,Y,cv=rfold)
print(result.mean())
print(result.std())
