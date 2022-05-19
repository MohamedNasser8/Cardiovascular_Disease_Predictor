import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('cardio_train.csv', sep=';')
df['age'] = df['age']/365.25
x = df[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
y = df[['cardio']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf = RandomForestClassifier(random_state = 0)
clf.fit(X_train,y_train.values.ravel())
y_pred = clf.predict(X_test)
a = accuracy_score(y_pred, y_test)
print(a)
model_filename = 'cardio-model.pkl'
print(clf.predict(x[:2]))
pickle.dump(clf, open(model_filename,'wb'))
print('Done')