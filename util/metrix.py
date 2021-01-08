import numpy as np
def get_metrix(y,y_pred):
    #accuracy、precision、recall、f1
    tp=0.0
    fp=0.0
    fn=0.0
    tn=0.0
    for i in range(len(y)):
        if y_pred[i]==1:
            if y[i]==1:
                tp+=1
            else:
                fp+=1
        else:
            if y[i]==y_pred[i]:
                tn+=1
            else:
                fn+=1
    accuracy=(tp+tn)/len(y)
    if tp+fp==0:
        precision=0
    else:
        precision=tp/(tp+fp)
    if tp+fn==0:
        recall=0
    else:
        recall=tp/(tp+fn)
    if precision<1e-6 or recall<1e-6:
        f1=0
    else:
        f1=2/(1/precision+1/recall)
    if tp+fn==0:
        tpr=0
    else:
        tpr=tp/(tp+fn)
    if tn+fp==0:
        fpr=0
    else:
        fpr=fp/(tn+fp)
    str1=""
    str1+=("tp:" + str(tp)+"\n")
    str1+=("fp:" + str(fp)+"\n")
    str1+=("fn:" + str(fn)+"\n")
    str1+=("tn:" + str(tn)+"\n")
    str1+=("accuracy:" + str(accuracy)+"\n")
    str1+=("precision:" + str(precision)+"\n")
    str1+=("recall:" + str(recall)+"\n")
    str1+=("f1:" + str(f1)+"\n")
    str1+=("tpr:" + str(tpr)+"\n")
    str1+=("fpr:" + str(fpr)+"\n")
    dict={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'accuracy':accuracy,'precision':precision,
          'recall':recall,'f1':f1,'tpr':tpr,'fpr':fpr,'to_string':str1}
    return dict


def getdicts(y,y_pre):
    tmp=np.copy(y_pre)
    np.sort(tmp)
    dicts=[]
    for thresh in tmp:
        tmp=[]
        for y_ in y_pre:
            if y_<=thresh:
                tmp.append(0)
            else:
                tmp.append(1)
        tmp=np.array(tmp)
        dicts.append(get_metrix(y,tmp))
    return dicts