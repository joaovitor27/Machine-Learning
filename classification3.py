import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

SEED = 20

URL = ('https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw'
       '/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv')

data = pd.read_csv(URL)
data.head()

sns.scatterplot(x='expected_hours', y='price', data=data, hue='unfinished')
sns.relplot(x='expected_hours', y='price', data=data, hue='unfinished', col='unfinished')
plt.show()

x = data[['expected_hours', 'price']]
y = data[['unfinished']]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)

model = LinearSVC()
model.fit(train_x, train_y)

predicts = model.predict(test_x)
accuracy = accuracy_score(predicts, test_y) * 100

print("Taxa de acerto: %.2f%%" % accuracy)

prevision_de_base = np.ones(540)
acuracia = accuracy_score(test_y, prevision_de_base) * 100
print("Taxa de acerto base: %.2f%%" % acuracia)
