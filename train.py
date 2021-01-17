import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
import tensorflow as tf
import cv2
import os
import datetime
from sklearn.metrics import f1_score,auc,roc_curve,accuracy_score,recall_score,precision_score,roc_auc_score

datapath="G:/lzhLaser/laserFilm/data"

#得到当前时间
def get_time():
    time= datetime.datetime.now()
    ft=datetime.datetime.strftime(time, '%Y-%m-%d %H:%M:%S')
    return ft

#将文件路径下的包含图片的文件夹里的图片进行resize并保存到resize文件夹下
def resize(imagePath):
    targetpath=imagePath+"/resize/"
    p=0
    for root,subs,file in os.walk(imagePath):
        if len(subs)==0:
            label=root.split("/")[-1]
            targetdir=targetpath+label
            os.makedirs(targetdir)
            for f in file:
                if f.split(".")[-1]=='jpg':
                    p+=1
                    image=cv2.imread(os.path.join(root,f))
                    image=cv2.resize(image,(224,224))
                    cv2.imwrite(targetdir+"/"+f,image)
                    print(p)



#从图片路径中读取包并且转化为X，Y形式的向量
def readimage(path,label):
    p = 0
    X=[]
    Y=[]
    for root, subs, file in os.walk(path):
        if len(subs) == 0:
            for f in file:
                if f.split(".")[-1] == 'jpg':
                    image = cv2.imread(os.path.join(root, f))
                    image=image/255
                    image=image.reshape(3,224,224)
                    X.append(image)
                    Y.append(label[p])
            p+=1
    X=np.array(X)
    Y=np.array(Y)
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)
    return X,Y


#打印当前模型的每层冻结情况,包括thresh
def printModel(model,update,thresh):
    for i,layer in enumerate(model.layers):
        if update:
            print(i,layer.name,'trainable:True')
        else:
            if(i<=thresh):
                layer.trainable = False
                print(i,layer.name,'trainable:False')
            else:
                print(i, layer.name, 'trainable:True')
#返回改进的vgg19网络模型
def vgg19(update=False):
    vgg19=tf.keras.applications.vgg19.VGG19(weights='imagenet', input_shape=(3,224, 224), include_top=False)
    x = vgg19.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dense(1024, activation='relu', name='fc2')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='predictions')(x)
    myvgg=tf.keras.models.Model(inputs=vgg19.inputs, outputs=x)
    thresh=-1
    if update == False:
        thresh=21
    printModel(myvgg,update,thresh)
    return myvgg

def iv3(update=False):
    iv3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', input_shape=(3,224, 224), include_top=False)
    x = iv3.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dense(1024, activation='relu', name='fc2')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='predictions')(x)
    myiv3 = tf.keras.models.Model(inputs=iv3.inputs, outputs=x)
    thresh = -1
    if update == False:
        thresh = len(iv3.layers)-1
    printModel(myiv3, update, thresh)
    return myiv3

def densenet121(update=False):
    densenet = tf.keras.applications.densenet.DenseNet121(weights='imagenet', input_shape=(3,224, 224),include_top=False)
    x = densenet.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dense(1024, activation='relu', name='fc2')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='predictions')(x)
    mydense = tf.keras.models.Model(inputs=densenet.inputs, outputs=x)
    thresh = -1
    if update == False:
        thresh = len(densenet.layers)-1
    printModel(mydense, update, thresh)
    return mydense

def resnet50(update=False):
    resnet = tf.keras.applications.resnet50.ResNet50(weights='imagenet', input_shape=(3,224, 224),include_top=False)
    x = resnet.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dense(1024, activation='relu', name='fc2')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='predictions')(x)
    myres = tf.keras.models.Model(inputs=resnet.inputs, outputs=x)
    thresh = -1
    if update == False:
        thresh = len(resnet.layers)-1
    printModel(myres, update, thresh)
    return myres



#epoch,train_loss,vali_loss,test_loss,train_acc,vali_acc,test_acc,test_auc,test_f1,test_recall,test_precision
def saveDataToTxt(path,title,data):
    f=open(path, 'a+')
    tt=get_time()+' | '+title+'\n'
    f.write(tt)
    table_head="epoch train_loss vali_loss test_loss train_acc vali_acc test_acc test_auc test_f1 test_recall test_precision train_time test_time\n"
    f.write(table_head)
    lens=len(data[0])
    for i in range(lens):
        tmp=""
        for item in data:
            tmp+=str(item[i])+" "
        f.write(tmp)
        f.write("\n")
    f.close()

def warpMycall(mycall):
    data=[]
    data.append(mycall.epoch)
    data.append(mycall.train_loss)
    data.append(mycall.vali_loss)
    data.append(mycall.test_loss)
    data.append(mycall.train_acc)
    data.append(mycall.vali_acc)
    data.append(mycall.test_acc)
    data.append(mycall.test_auc)
    data.append(mycall.test_f1)
    data.append(mycall.test_recall)
    data.append(mycall.test_precision)
    data.append(mycall.train_time)
    data.append(mycall.test_time)
    return data

class mycallback(tf.keras.callbacks.Callback):
    def __init__(self,test_x,test_y,weights_savedir,kf):
        self.test_feature=test_x
        self.test_y=test_y
        self.weights_savedir=weights_savedir
        self.epoch_begin_time=datetime.datetime.now()
        self.kf=kf
    def on_train_begin(self, logs={}):
        self.epoch=[]
        self.train_loss=[]
        self.vali_loss=[]
        self.test_loss=[]
        self.train_acc=[]
        self.vali_acc=[]
        self.test_acc=[]
        self.test_f1=[]
        self.test_auc=[]
        self.test_precision=[]
        self.test_recall=[]
        self.train_time=[]
        self.test_time=[]

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin_time=datetime.datetime.now()

    def on_epoch_end(self, epoch, logs={}):
        cur=datetime.datetime.now()
        dur=cur-self.epoch_begin_time
        self.train_time.append(dur.microseconds)
        if epoch>35:
            self.model.save_weights(self.weights_savedir+"/"+"kf-"+str(self.kf)+"epoch-"+("%d"%epoch)+".h5")
        self.epoch.append(epoch)
        self.train_acc.append(logs.get('binary_accuracy'))
        self.train_loss.append(logs.get('loss'))
        self.vali_acc.append(logs.get('val_binary_accuracy'))
        self.vali_loss.append(logs.get('val_loss'))
        begin=datetime.datetime.now()
        test_y_pred=self.model.predict(self.test_feature)
        end=datetime.datetime.now()
        during=end-begin
        self.test_time.append(during.microseconds)
        test_y_pred=test_y_pred.reshape(-1)
        test_acc=tf.keras.metrics.binary_accuracy(self.test_y,test_y_pred)
        test_loss=tf.keras.losses.binary_crossentropy(self.test_y,test_y_pred)
        self.test_acc.append(np.float(test_acc))
        self.test_loss.append(np.float(test_loss))
        roc_auc = roc_auc_score(self.test_y,test_y_pred)
        self.test_auc.append(roc_auc)
        test_y_pre=[]
        for y_pred in test_y_pred:
            if y_pred<0.5:
                test_y_pre.append(0)
            else:
                test_y_pre.append(1)
        test_y_pre=np.array(test_y_pre)
        #print("te" + str(accuracy_score(self.test_y, test_y_pre)))
        self.test_f1.append(f1_score(self.test_y,test_y_pre))
        self.test_precision.append(precision_score(self.test_y,test_y_pre))
        self.test_recall.append(recall_score(self.test_y,test_y_pre))



def switchModel(model_name,update=False):
    if model_name == 'vgg19':
        return vgg19(update)
    elif model_name=='resnet50':
        return resnet50(update)
    elif model_name=='densenet121':
        return densenet121(update)
    elif model_name=='iv3':
        return iv3(update)
    else:
        raise ValueError("no model")
def train(model_name,X,Y,weights_save_dir,data_save_dir,epoch):
    kf = StratifiedKFold(n_splits=5,random_state=0)
    p=1
    for i in range(2):
        if i==0:
            update=False
            wsd=weights_save_dir+'/freeze/'+model_name
            dsp=data_save_dir+'/freeze/'+model_name+'.txt'
        else:
            update=True
            wsd=weights_save_dir+'/update/'+model_name
            dsp=data_save_dir+'/update/'+model_name+'.txt'
        for train_index, test_index in kf.split(X,Y):
            model=switchModel(model_name,update)
            model.summary()
            model.compile(loss=tf.keras.losses.binary_crossentropy,
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          metrics=[tf.keras.metrics.binary_accuracy])
            title="add fc layer "+"number: "+str(p)+" fold"
            print(title)
            train_x, vali_x,train_y,vali_y= train_test_split(X[train_index],Y[train_index],test_size=0.15,random_state=0)
            testX, testY = X[test_index], Y[test_index]
            mycall=mycallback(testX,testY,wsd,p)
            model.fit(train_x,train_y,epochs=epoch,shuffle=True,validation_data=(vali_x, vali_y), callbacks=[mycall])
            data=warpMycall(mycall)
            saveDataToTxt(dsp,title,data)
            p+=1

def model_train(modelname):
    X,Y=readimage(datapath,[0,1])
    train(modelname,X,Y,'weights','data',40)

if __name__=='__main__':
    model_train('vgg19')
    model_train('densenet121')
    model_train('resnet50')
    model_train('iv3')



