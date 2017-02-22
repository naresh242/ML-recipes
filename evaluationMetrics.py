from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataframe=read_csv(filename,names=names)
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]

# classification accuracy
seed=7
nfold=10
kfold=KFold(n_splits=nfold,random_state=seed)
model=LogisticRegression()
scoring='accuracy'
result=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
print(result)
print(result.mean())
print(result.std())

# Cross Validation Classification LogLoss
seed=7
nfold=10
kfold=KFold(n_splits=nfold,random_state=seed)
model=LogisticRegression()
scoring='neg_log_loss'
result=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
print(result)
print(result.mean())
print(result.std())

# Cross Validation Classification AUC (Area under ROC)
seed=7
nfold=10
kfold=KFold(n_splits=nfold,random_state=seed)
model=LogisticRegression()
scoring='roc_auc'
result=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
print(result)
print(result.mean())
print(result.std())

#confusion matrix
seed=7
nfold=10
test_size=0.33
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)
model=LogisticRegression()
model.fit(X_train,Y_train)
preditions=model.predict(X_test)
mat=confusion_matrix(Y_test,preditions)
print(mat)

#classification report
seed=7
nfold=10
test_size=0.33
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)
model=LogisticRegression()
model.fit(X_train,Y_train)
preditions=model.predict(X_test)
mat=classification_report(Y_test,preditions)
print(mat)

# Cross Validation Regression MAE
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)


# Cross Validation Regression MSE
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

# Cross Validation Regression R^2
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = 'r2'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)




