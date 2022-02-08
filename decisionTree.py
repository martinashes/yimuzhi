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
    """	for i in range(1, 16):
        for j in range(2, 20):"""
    dataArr, classLabels = loadDataSet('trainingdata_20220208.txt')
    testArr, testLabelArr = loadDataSet('testdata_20220208.txt')
    bdt = DecisionTreeClassifier(min_samples_leaf=2, max_depth=1, max_leaf_nodes=2)
    decision_tree = bdt.fit(dataArr, classLabels)
    r = export_text(decision_tree, decimals=8,
                    feature_names=['stars', 'live_days', 'live_times', 'live_hours', 'revenue', 'pcu',
                                   'audience', 'eff_audience', 'acu', 'watch_time', 'eff_watchtime',
                                   'interact_pll', 'interact_times', 'new_fans', 'fans', 'old_fans'])
    print(r)
    predictions = bdt.predict(dataArr)

    errArr = np.mat(np.ones((len(dataArr), 1)))
    traing_error_rate = float(errArr[predictions != classLabels].sum() / len(dataArr) * 100)
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != classLabels].sum() / len(dataArr) * 100))
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    test_error_rate = float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100)
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100))
    # print(i, j, traing_error_rate, test_error_rate)

    dot_data = tree.export_graphviz(decision_tree, out_file=None,  # 绘制决策树
                                    feature_names=['stars', 'live_days', 'live_times', 'live_hours', 'revenue', 'pcu',
                                                   'audience', 'eff_audience', 'acu', 'watch_time', 'eff_watchtime',
                                                   'interact_pll', 'interact_times', 'new_fans', 'fans', 'old_fans'],
                                    filled=True, rounded=True,
                                    class_names=['drop', 'keep', 'raise', '5stars'],
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('iris.pdf')
