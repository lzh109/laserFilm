import numpy as np
def saveTxt(file,a):
    p=0
    for row in a:
        s=""
        for i in range(len(row)-1):
            stmp="%.18e"%row[i]
            s=s+stmp+" "
        stmp="%.18e"%row[len(row)-1]
        s=s+stmp+"\n"
        file.write(s)
        p+=1
        print("%d done"%p)

def getdata_fromtxt(txtPath):
    data=np.loadtxt(txtPath)
    features=[]
    labels=[]
    for da in data:
        features.append(da[:-1])
        labels.append(da[-1])
    features=np.array(features)
    labels=np.array(labels)
    return features,labels
