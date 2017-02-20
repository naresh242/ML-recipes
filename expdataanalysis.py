# Load CSV using Pandas from URL
from pandas import read_csv
url = ''
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(url, names=names)
print(data.shape)


filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
print(data.shape)

# descriptive stats
pandas.set_option('display.width', 50)
pandas.set_option('precision', 3)
description=data.describe()
print(description)

#labels are balanced or not
data.groupby('class').size()

#correlation matrix
data.corr(method='pearson')

#correlation matrix
data.corr(method='pearson')

#check skewness
data.skew()

# histograms
data.hist()
plt.show()

# density plots
data.plot(kind='density',subplots=True,layout=(3,3),sharex=False)
plt.show()

# box Whisker plots 
data.plot(kind='box',subplots=True,layout=(3,3),sharex=False,sharey=False)
plt.show()


#correlation matrix
correlations=data.corr(method='pearson')
print(correlations)
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=-1,vmax=1)
ticks = numpy.arange(0,9,1)
fig.colorbar(cax)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


#scatter matrix
pandas.scatter_matrix(data)
plt.show()
