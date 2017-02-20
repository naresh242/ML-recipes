import scipy
import numpy
import matplotlib.pyplot as plt
import sklearn
import pandas

#line plot
na=numpy.array([5,7,1])

plt.plot(na)
plt.xlabel('some x axis')
plt.ylabel('some y axis')
plt.show()

#scatter plot

nx=numpy.array([5,7,1])
ny=numpy.array([5,7,1])

plt.scatter(nx,ny)
plt.xlabel('some x axis')
plt.ylabel('some y axis')
plt.show()

#series
nx=numpy.array([5,7,1])
labels=['a','b','c']
mseries=pandas.Series(nx,index=labels)

#dataframes
na=[[1,2,3,4],[3,4,5,6],[3,5,7,8],[1,2,3,4]]
xlab=[1,2,3,4]
ylab=[1,2,3,4]
dfm=pandas.DataFrame(na,index=xlab,columns=ylab)
print(dfm)

