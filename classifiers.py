from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from joblib import dump


data = pd.read_csv("ph-data.csv")


X = pd.DataFrame(columns=['col1', 'col2', 'col3'])
X['col1'], X['col2'], X['col3'] = data['red'], data['green'], data['blue']

y = pd.DataFrame()
y['col1'] = data['label']

y.loc[y['col1'] < 6, 'col1'] = -1
y.loc[y['col1'] == 6, 'col1'] = 0
y.loc[y['col1'] == 7, 'col1'] = 0
y.loc[y['col1'] == 8, 'col1'] = 0
y.loc[y['col1'] == 9, 'col1'] = 0
y.loc[y['col1'] > 9, 'col1'] = 1



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=5)
y_new_train = y_train['col1'].values


#KNN
neighbors = KNeighborsClassifier(n_neighbors=5, p=2)
neighbors.fit(X_train, y_new_train)
y_pred = neighbors.predict(X_test)
print(accuracy_score(y_test, y_pred))


#GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_new_train)
y_pred = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred))


#DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=5, max_depth=4)
tree.fit(X_train, y_new_train)
y_pred = tree.predict(X_test)
print(accuracy_score(y_test, y_pred))


#Full pack learn on GaussNB
y_new = y['col1'].values
# model = GaussianNB()
model = DecisionTreeClassifier(random_state=5, max_depth=5)
# model = KNeighborsClassifier(n_neighbors=5, p=2)
model.fit(X, y_new)

#r g b
X_new = [[255, 38, 0],[255, 124, 0],[141, 250, 0],[179, 68, 198],[111, 43, 142]]

df_new = pd.DataFrame(X_new, columns=['col1', 'col2', 'col3'])

print(df_new.head())

y_pred = model.predict(df_new)

print(y_pred)

dump(model, 'chekin_model.mdl')
