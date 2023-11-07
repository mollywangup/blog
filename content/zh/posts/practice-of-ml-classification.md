---
title: "机器学习实践 - 垃圾邮件分类器"
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
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def runplt():
    '''
    绘制 ROC 曲线的准备工作
    '''
    fig, ax = plt.subplots()
    ax.set_title('ROC curve and AUC')
    ax.set_xlabel('FPR (False Positive Rate)')
    ax.set_ylabel('TPR (True Positive Rate)')
    ax.plot([0, 1], [0, 1], color='navy', ls='--', label='random: 0.5')
    ax.plot([0, 0, 1, 1], [0, 1, 1, 1], color='forestgreen', ls='--', label='perfect: 1')
    return ax

def main():
    '''
    任务：垃圾邮件分类
    '''
    # 加载数据集
    df = pd.read_table('/path/to/SMSSpamCollection.txt', header=None)
    feature, target = df[1], df[0]
    
    # 拆分训练集
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=777)
    print('>>> 训练集：{} 组, 测试集：{} 组 <<<'.format(X_train.shape[0], X_test.shape[0]))
    
    # 预处理: 处理文本 (词袋模型)
    cv = CountVectorizer(stop_words='english')
    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # 构建多个分类器
    names = [
        'KNN', 
        'Logistic', 
        'SVM',
        'NaiveBayes', 
        'DecisionTree', 
        'RandomForest', 
        'XGBoost',
    ]
    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        LogisticRegression(),
        SVC(kernel='linear', probability=True),
        MultinomialNB(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        xgb.XGBClassifier(tree_method='hist'),
    ]
    
    # 批处理: 模型训练, 保存评估指标, 绘制 ROC 曲线
    ax = runplt()
    report = []
    
    for name, clf in zip(names, classifiers):
        # 模型训练
        start_time = time.time()
        clf.fit(X_train, y_train)
        duration = time.time() - start_time
        
        # 模型评估: 拟合度
        score_train = clf.score(X_train, y_train)
        score_test = clf.score(X_test, y_test)
        print('{} (耗时 {:.5f} 秒):\n  训练集准确率: {:.3f}\n  测试集准确率: {:.3f}'.format(name, duration, score_train, score_test))
        
        # 模型评估: 准确率/精确率/召回率/F1
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 模型评估: 绘制 ROC 曲线并标注 AUC 值
        y_score = clf.predict_proba(X_test)
        if y_score.shape[1] == 2:
            y_score = y_score[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        ax.plot(fpr, tpr, label='{}: ${:.3f}$'.format(name, auc))

        # 将所有评估指标保存在 dataframe
        _ = {
            'classifier': name, 
            'duration': duration, 
            'accuracy_train': score_train, 
            'accuracy_test': score_test, 
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'AUC': auc,
        }
        report.append(_)

    # 打印 dataframe
    df = pd.DataFrame(report)
    df = df.sort_values(by='AUC', ascending=False)
    print(df)
    
    # 保存 ROC 曲线
    ax.legend(loc='best', frameon=True, fontsize='small')
    plt.savefig('ROC.svg')
    plt.show()
 

if __name__ == '__main__':
    main()
```

## 运行结果

指标意义详见：<a href="https://mollywangup.com/posts/notes-machine-learning/#%E5%88%86%E7%B1%BB%E6%8C%87%E6%A0%87" target="_blank">分类指标</a>

<img src='https://user-images.githubusercontent.com/46241961/280677677-52b66dee-1fe1-4523-ad5f-ba1e35f44a78.png' alt='打印'>

<br><img src='https://user-images.githubusercontent.com/46241961/280677743-c441f94b-81f2-43ee-b3c5-e8f9136f5e97.svg' alt='ROC'>
