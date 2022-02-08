# -*-coding:utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_text
import pydotplus
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from IPython.display import Image


def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat, labelMat


if __name__ == '__main__':
    for i in range(1, 31, 3):
        for j in range(2, 32, 3):
            dataArr, classLabels = loadDataSet('trainingdata_20220208.txt')
            testArr, testLabelArr = loadDataSet('testdata_20220208.txt')
            bdt = DecisionTreeClassifier(min_samples_leaf=5, max_depth=i, max_leaf_nodes=j)
            decision_tree = bdt.fit(dataArr, classLabels)
            predictions = bdt.predict(dataArr)
            errArr = np.mat(np.ones((len(dataArr), 1)))
            traing_error_rate = float(errArr[predictions != classLabels].sum() / len(dataArr) * 100)
            predictions = bdt.predict(testArr)
            errArr = np.mat(np.ones((len(testArr), 1)))
            test_error_rate = float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100)
            print(i, j, traing_error_rate, test_error_rate)
