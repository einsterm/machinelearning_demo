# -*- encoding=utf-8 -*-
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def test():
    dataSet = loadtxt("testSet.txt", delimiter=",")
    X = dataSet[:, 0:8]
    Y = dataSet[:, 8]

    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("准确率:%.2f%%" % (accuracy * 100.0))

if __name__ == '__main__':
    test()