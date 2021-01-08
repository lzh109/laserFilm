from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,\
    f1_score
import forward as fo
import util.metrix as mt
import util.txtutil as tu
# path="/Users/lizhenhao/Desktop/helloworld/im"



#confusion_matrix
#                      预测值
#               正样本        负样本
# 实际值  正样本    tp           fn
#        负样本    fp           tn


def svm_train():
    train_feature,train_label=tu.getdata_fromtxt("/Users/lizhenhao/Desktop/helloworld/im/vgg/train.txt")
    test_feature,test_label=tu.getdata_fromtxt("/Users/lizhenhao/Desktop/helloworld/im/vgg/test.txt")
    svm=SVC().fit(train_feature,train_label)
    test_label_pred=svm.predict(test_feature)
    dict=mt.get_metrix(test_label,test_label_pred)
    print(dict['to_string'])
    # true = np.sum(test_label_pred == test_label)
    # print('预测对的结果数目为：', true)
    #
    # print('预测错的的结果数目为：', test_label.shape[0])
    #
    # print('预测错的的结果数目为：', test_label.shape[0] - true)
    #
    # print('预测结果准确率为：', true / test_label.shape[0])
    #
    # print('使用SVM预测breast_cancer数据的准确率为：',
    #
    #       accuracy_score(test_label, test_label_pred))
    #
    # print('使用SVM预测breast_cancer数据的精确率为：',
    #
    #       precision_score(test_label, test_label_pred))
    #
    # print('使用SVM预测breast_cancer数据的召回率为：',
    #
    #       recall_score(test_label, test_label_pred))
    #
    # print('使用SVM预测breast_cancer数据的F1值为：',
    #
    #       f1_score(test_label, test_label_pred))
    # print('使用SVM预测iris数据的分类报告为：', '\n',
    #       classification_report(test_label,
    #
    #                             test_label_pred))
if __name__=='__main__':
    svm_train()