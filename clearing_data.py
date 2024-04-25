import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
col_names = iris.feature_names
print(col_names)

X = iris.data
y = iris.target

sepalLength = X[:,0] #sepal - чашелистик
sepalWidth = X[:,1]
petalLength = X[:,2]
petalWidth = X[:,3]


def GetWhiskers(column):
    '''
    Написать функцию, которая будет возвращать значение "усов", отстоящих на 3 IQR от Q1 и Q3
    '''
    times = 3
    q1q2q3 = np.quantile(column, [0.25,0.50,0.75])
    IQR = q1q2q3[2]-q1q2q3[0]
    w1 = q1q2q3[0]-times*IQR
    w2 = q1q2q3[2]+times*IQR
    return (q1q2q3[0], column[column >= w1].min()), (q1q2q3[2], column[column <= w2].max())

sl, sw, pl, pw = GetWhiskers(sepalLength), GetWhiskers(sepalWidth), GetWhiskers(petalLength), GetWhiskers(petalWidth),
print(sl, sw, pl, pw)


df = pd.DataFrame(data=X[:,:],columns=['col1', 'col2', 'col3', 'col4'])

df['col5'] = y[:]

print(len(df))
array = [sl, sw, pl, pw]

for i in range(4):
    arr1 = df.loc[df['col{}'.format(i + 1)] > array[i][1][1]]

    for j in range(len(arr1)):
        if len(arr1) != 0:
            df = df.drop(arr1.index.values[j])


for i in range(4):
    arr2 = df.loc[df['col{}'.format(i + 1)] < array[i][0][1]]

    for j in range(len(arr2)):
        if len(arr2) != 0:
            df = df.drop(arr2.index.values[j])



X = pd.DataFrame()
y = pd.DataFrame()

X['col1'], X['col2'], X['col3'], X['col4'] = df['col1'], df['col2'], df['col3'], df['col4']
y['col1'] = df['col5']


X_train, X_test, y_train, y_test = train_test_split(X, y['col1'].values, test_size=0.7, random_state=1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred))
