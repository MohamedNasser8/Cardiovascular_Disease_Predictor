import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("cardio_train.csv", sep=';')
df['age'] = df['age']/365.25
x = df[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
y = df[['cardio']]
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train.values.ravel())
y_pred = clf.predict(X_test)
a = accuracy_score(y_pred, y_test)
print(a)
model_filename = 'cardio-model.pkl'
print(clf.predict(x[:2]))
pickle.dump(clf, open(model_filename, 'wb'))
cardio_count=df["cardio"].value_counts()
print(cardio_count)
print(" The Number of people with cardiovascular disease is: %s"%cardio_count.get(1),"\n", "The Number of people without cardiovascular disease is: %s" % cardio_count.get(0))
figure1 = plt.figure('Distribution of Patients with CardioVascular Disease in the dataset',figsize=(10,10))
GraphColors = ['blue','Red']
graph = plt.bar(["Without CardioVascular Disease","With CardioVascular Disease"],cardio_count, color = GraphColors)
plt.title('Distribution of Patients with CardioVascular Disease in the dataset')
Cardio_percentages = df["cardio"].value_counts()/(df["cardio"].count())*100
i = 0
for j in graph:
    width = j.get_width()
    height = j.get_height()
    x, y = j.get_xy()
    print(type(height))
    plt.text(x+width/2, y+height*1.01, str(Cardio_percentages[i])+'%', ha='center', weight='bold')
    i += 1
plt.show()
print('Done Compiling')