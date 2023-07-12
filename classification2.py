import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

SEED = 20

URL = ('https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw'
       '/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv')

data = pd.read_csv(URL)
data.head()
x = data[['home', 'how_it_works', 'contact']]
y = data[['bought']]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)

model = LinearSVC()
model.fit(train_x, train_y)

predicts = model.predict(test_x)
accuracy = accuracy_score(predicts, test_y) * 100

print("Taxa de acerto: %.2f%%" % accuracy)

