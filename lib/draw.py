import itertools
import matplotlib.pyplot as plt
import numpy as np


def decode_one_hot(true_label, pred_label):
    ''' 返回one-hot解码后的标签    '''
    y_true = np.argmax(true_label, axis=1)
    y_pred = np.argmax(pred_label, axis=1)
    return y_true, y_pred


def get_confusion_matrix_figure(cm, labels=('(+1)', '(0)', '(-1)'), percentage=False, cmap=plt.cm.Blues):
    """
    - cm : 需要显示的混淆矩阵
    - classes : 混淆矩阵中每一行每一列对应的类别
    - percentage : True:显示百分比（保留两位小数）, False:显示个数
    """
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    if percentage:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ticks = np.arange(len(labels))
    plt.xticks(ticks=ticks, labels=labels)
    plt.yticks(ticks=ticks, labels=labels)
    fmt = '.2%' if percentage else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i][j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


# 处理分类报告的原始数据
def deal_classification_report(original_report):
    report = []
    for line in original_report.strip().split('\n')[2:]:
        s = line.strip().split('   ')
        if s[0] == 'accuracy':
            tmp = s[-1 - 3:-1]
        else:
            tmp = s[1:4]
        if tmp:
            report.append(tmp[:])

    # 交换平均值和准确率的位置
    tmp = report[4]
    report[4] = report[3]
    report[3] = tmp

    return report


def get_stock_data(label, last):
    # 股票不涨不跌的标记位，本函数的股票比例为  涨：不涨不跌：跌 = 1：3：1
    flag = False
    stocks = [last]

    for i in range(len(label)):
        if label[i] == 1:
            last *= 1.002
        elif label[i] == 3:
            last *= 0.998
        else:
            flag = True
        stocks.append(last)
        if flag:
            stocks.append(last)
            stocks.append(last)
            stocks.append(last)

    return stocks
