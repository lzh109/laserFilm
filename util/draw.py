import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import util.metrix
import numpy as np


def draw_roc (y_,y_preds,name_list,savepath):
     ###计算真正率和假正率
    lw = 2
    plt.figure(figsize=(10, 10))
    colors=['green','darkorange','blue','red']
    p=0
    for i in range(len(y_)):
        y=y_[i]
        y_pred=y_preds[i]
        fpr,tpr,thresh=roc_curve(y,y_pred)
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        plt.plot(fpr, tpr, color=colors[p],
             lw=lw, label=name_list[p]+' (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        p+=1
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()


def draw_acc(train_acc,vali_acc,test_acc,epoch,savepath):
    plt.figure(figsize=(20, 10))
    train_acc_new=[0]
    [train_acc_new.append(ans) for ans in train_acc]
    train_acc_new=np.array(train_acc_new)
    vali_acc_new = [0]
    [vali_acc_new.append(ans) for ans in vali_acc]
    vali_acc_new = np.array(vali_acc_new)
    test_acc_new = [0]
    [test_acc_new.append(ans) for ans in test_acc]
    test_acc_new = np.array(test_acc_new)
    epoch_new = [0]
    [epoch_new.append(ans+1) for ans in epoch]
    epoch_new = np.array(epoch_new)
    plt.plot(epoch_new,train_acc_new,color='green',lw=2,label='train_data accuracy')
    for i_x, i_y in zip(epoch_new, train_acc_new):
        plt.text(i_x, i_y, '({})'.format(i_y))
    plt.plot(epoch_new,vali_acc_new,color='blue',lw=2,label='validation_data accuracy')
    for i_x, i_y in zip(epoch_new, vali_acc_new):
        plt.text(i_x, i_y, '({})'.format(i_y))
    plt.plot(epoch_new,test_acc_new,color='red',lw=2,label='test_data accuracy')
    for i_x, i_y in zip(epoch_new, test_acc_new):
        plt.text(i_x, i_y, '({})'.format(i_y))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([0, np.max(epoch)])
    plt.ylim([0.0, 1.05])
    plt.title('Accuray')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()
def draw_loss(train_loss,vali_loss,test_loss,epoch,savepath):
    plt.figure(figsize=(20, 10))
    plt.plot(epoch, train_loss, color='green', lw=2, label='train_data loss')
    for i_x, i_y in zip(epoch, train_loss):
        plt.text(i_x, i_y, '({})'.format(i_y))
    plt.plot(epoch, vali_loss, color='blue', lw=2, label='validation_data loss')
    for i_x, i_y in zip(epoch, vali_loss):
        plt.text(i_x, i_y, '({})'.format(i_y))
    test_loss=np.array(test_loss)
    plt.plot(epoch, test_loss, color='red', lw=2, label='test_data loss')
    for i_x, i_y in zip(epoch, test_loss):
        plt.text(i_x, i_y, '({})'.format(i_y))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([0, np.max(epoch)])
    plt.ylim([0.0, 1.05])
    plt.title('Loss')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    plt.show()