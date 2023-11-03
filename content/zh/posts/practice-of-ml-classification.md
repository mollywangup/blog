---
title: "机器学习实践 - 垃圾邮件分类"
date: 2023-10-12T10:20:28Z
draft: false
description: 对比 KNN、逻辑回归、SVM、朴素贝叶斯、决策树、随机森林、XGBoost 等多个分类器来对垃圾邮件进行分类。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Machine Learning
- sklearn
- Classification
categories:
- Practice
libraries:
- mathjax
---

数据源：<a href="https://archive.ics.uci.edu/dataset/228/sms+spam+collection" target="_blank">SMS Spam Collection</a>

## 代码

```python
import time
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def main():
    '''
    任务：垃圾邮件分类
    '''
    # 加载数据集
    df = pd.read_table('/path/to/SMSSpamCollection.txt', header=None)
    feature, target = df[1], df[0]
    
    # 拆分训练集
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=777)
    
    # 处理文本和标签
    cv = CountVectorizer(stop_words='english')
    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # 构建多个分类器
    names = ['K 近邻', 'LogisticRegression', 'Naive Bayes', 'DecisionTree', 'RandomForest', 'XGBoost']
    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        LogisticRegression(),
        MultinomialNB(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        xgb.XGBClassifier(tree_method='hist'),
    ]
    
    # 批量训练
    for name, clf in zip(names, classifiers):
        start_time = time.time()
        clf.fit(X_train, y_train)
        duration = time.time() - start_time
        score_train = clf.score(X_train, y_train)
        score_test = clf.score(X_test, y_test)
        # con = confusion_matrix()
        print('{} (耗时 {:.5f}):\n  训练集准确率: {:.3f}\n  测试集准确率: {:.3f}'.format(name, duration, score_train, score_test))


if __name__ == '__main__':
    main()
```
