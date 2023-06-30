from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

pig1 = [0, 1, 0]
pig2 = [0, 1, 1]
pig3 = [1, 1, 0]

dog1 = [0, 1, 1]
dog2 = [1, 0, 1]
dog3 = [1, 1, 1]

train_x = [pig1, pig2, pig3, dog1, dog2, dog3]
train_y = [1, 1, 1, 0, 0, 0]

model = LinearSVC()
model.fit(train_x, train_y)


if __name__ == "__main__":
    animal = [1, 1, 1]
    animal2 = [1, 1, 0]
    animal3 = [0, 1, 1]
    predicts = model.predict([animal, animal2, animal3])
    corrects = [0, 1, 1]
    matches = (predicts == corrects).sum()
    taxa_acerto = accuracy_score(corrects, predicts)
    print("Taxa de acerto: %.2f" % (taxa_acerto * 100))
