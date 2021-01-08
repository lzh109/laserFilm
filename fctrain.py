import tensorflow as tf
from models import fc
import numpy as np
from util import draw
from util import metrix
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def getdata_fromtxt(txtPath):
    data=np.loadtxt(txtPath)
    np.random.shuffle(data)
    features=[]
    labels=[]
    for da in data:
        features.append(da[:-1])
        if da[-1]==1:
            labels.append(1)
        else:
            labels.append(0)
    features=np.array(features)
    labels=np.array(labels)
    return features,labels

class mycallback(tf.keras.callbacks.Callback):
    def __init__(self,test_f,test_y,weights_savedir):
        self.test_feature=test_f
        self.test_y=test_y
        self.weights_savedir=weights_savedir
    def on_train_begin(self, logs={}):
        self.epoch=[]
        self.train_loss=[]
        self.vali_loss=[]
        self.test_loss=[]
        self.train_acc=[]
        self.vali_acc=[]
        self.test_acc=[]
    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(self.weights_savedir+"/"+"epoch-"+("%d"%epoch)+".h5")
        self.epoch.append(epoch)
        self.train_acc.append(logs.get('binary_accuracy'))
        self.train_loss.append(logs.get('loss'))
        self.vali_acc.append(logs.get('val_binary_accuracy'))
        self.vali_loss.append(logs.get('val_loss'))
        test_y_pred=self.model.predict(self.test_feature)
        test_y_pred=test_y_pred.reshape(-1)
        test_acc=tf.keras.metrics.binary_accuracy(self.test_y,test_y_pred)
        test_loss=tf.keras.losses.binary_crossentropy(self.test_y,test_y_pred)
        self.test_acc.append(test_acc)
        self.test_loss.append(test_loss)
        # train_y_pred=self.model.predict(self.train_feature)
        # vali_y_pred=self.model.predict(self.vali_feature)
        # test_y_pred=self.model.predict(self.test_feature)
        # res_train=mt.get_metrix(self.train_y,train_y_pred)
        # res_vali = mt.get_metrix(self.vali_y, vali_y_pred)
        # res_test = mt.get_metrix(self.test_y, test_y_pred)
        # self.train_metrix.append(res_train)
        # self.vali_metrix.append(res_vali)
        # self.test_metrix.append(res_test)
def vggtrain():
    print("vgg begin")
    model=fc.get_model(25088)
    root='/Users/lizhenhao/Desktop/helloworld/im/vgg'
    weights_savedir="/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/vgg"
    acc_savepath="/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/vgg/acc.png"
    loss_savepath="/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/vgg/loss.png"
    train_f,train_y=getdata_fromtxt(root+'/train.txt')
    vali_f,vali_y=getdata_fromtxt(root+'/vali.txt')
    test_f,test_y=getdata_fromtxt(root+'/test.txt')
    mycall=mycallback(test_f,test_y,weights_savedir)
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.binary_accuracy])
    model.fit(train_f,train_y,epochs=1,shuffle=True,validation_data=(vali_f,vali_y),callbacks=[mycall])
    draw.draw_acc(mycall.train_acc,mycall.vali_acc,mycall.test_acc,mycall.epoch,acc_savepath)
    draw.draw_loss(mycall.train_loss,mycall.vali_loss,mycall.test_loss,mycall.epoch,loss_savepath)
    # 输出测试集最高的准确率
    print('max test accuracy:')
    print(mycall.test_acc)
    print('vgg done')
    return model

def densenettrain():
    print("densenet begin")
    model = fc.get_model(50176)
    root = '/Users/lizhenhao/Desktop/helloworld/im/densenet'
    weights_savedir = "/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/densenet"
    acc_savepath = "/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/densenet/acc.png"
    loss_savepath = "/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/densenet/loss.png"
    train_f, train_y = getdata_fromtxt(root + '/train.txt')
    vali_f, vali_y = getdata_fromtxt(root + '/vali.txt')
    test_f, test_y = getdata_fromtxt(root + '/test.txt')
    mycall = mycallback(test_f, test_y, weights_savedir)
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.binary_accuracy])
    model.fit(train_f, train_y, epochs=20, shuffle=True, validation_data=(vali_f, vali_y), callbacks=[mycall])
    draw.draw_acc(mycall.train_acc, mycall.vali_acc, mycall.test_acc, mycall.epoch, acc_savepath)
    draw.draw_loss(mycall.train_loss, mycall.vali_loss, mycall.test_loss, mycall.epoch, loss_savepath)
    print('densenet done')
    return model

def iv3train():
    print("iv3 begin")
    model = fc.get_model(51200)
    root = '/Users/lizhenhao/Desktop/helloworld/im/iv3'
    weights_savedir = "/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/iv3"
    acc_savepath = "/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/iv3/acc.png"
    loss_savepath = "/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/iv3/loss.png"
    train_f, train_y = getdata_fromtxt(root + '/train.txt')
    vali_f, vali_y = getdata_fromtxt(root + '/vali.txt')
    test_f, test_y = getdata_fromtxt(root + '/test.txt')
    mycall = mycallback(test_f, test_y, weights_savedir)
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.binary_accuracy])
    model.fit(train_f, train_y, epochs=20, shuffle=True, validation_data=(vali_f, vali_y), callbacks=[mycall])
    draw.draw_acc(mycall.train_acc, mycall.vali_acc, mycall.test_acc, mycall.epoch, acc_savepath)
    draw.draw_loss(mycall.train_loss, mycall.vali_loss, mycall.test_loss, mycall.epoch, loss_savepath)
    print('iv3 done')
    return model


def resnet50train():
    print("resnet50 begin")
    model = fc.get_model(2048)
    root = '/Users/lizhenhao/Desktop/helloworld/im/resnet50'
    weights_savedir = "/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/resnet50"
    acc_savepath = "/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/resnet50/acc.png"
    loss_savepath = "/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/resnet50/loss.png"
    train_f, train_y = getdata_fromtxt(root + '/train.txt')
    vali_f, vali_y = getdata_fromtxt(root + '/vali.txt')
    test_f, test_y = getdata_fromtxt(root + '/test.txt')
    mycall = mycallback(test_f, test_y, weights_savedir)
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.binary_accuracy])
    model.fit(train_f, train_y, epochs=20, shuffle=True, validation_data=(vali_f, vali_y), callbacks=[mycall])
    draw.draw_acc(mycall.train_acc, mycall.vali_acc, mycall.test_acc, mycall.epoch, acc_savepath)
    draw.draw_loss(mycall.train_loss, mycall.vali_loss, mycall.test_loss, mycall.epoch, loss_savepath)
    print('resnet50 done')
    return model


def plot_roc():
    y_preds = []
    y_=[]
    name_list = ['vgg+fc','densenet121+fc','iv3+fc','resnet50+fc']
    vggroot = "/Users/lizhenhao/Desktop/helloworld/im/vgg"
    vgg=fc.get_model(25088)
    vgg.load_weights("/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/vgg/epoch-19.h5")
    test_f, test_y = getdata_fromtxt(vggroot + '/test.txt')
    test_y_pred = vgg.predict(test_f)
    test_y_pred = test_y_pred.reshape(-1)
    y_preds.append(test_y_pred)
    y_.append(test_y)

    denseroot = "/Users/lizhenhao/Desktop/helloworld/im/densenet"
    den = fc.get_model(50176)
    den.load_weights("/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/densenet/epoch-19.h5")
    test_f, test_y = getdata_fromtxt(denseroot + '/test.txt')
    test_y_pred = den.predict(test_f)
    test_y_pred = test_y_pred.reshape(-1)
    y_preds.append(test_y_pred)
    y_.append(test_y)

    iv3root = "/Users/lizhenhao/Desktop/helloworld/im/iv3"
    iv3 = fc.get_model(51200)
    iv3.load_weights("/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/iv3/epoch-19.h5")
    test_f, test_y = getdata_fromtxt(iv3root + '/test.txt')
    test_y_pred = iv3.predict(test_f)
    test_y_pred = test_y_pred.reshape(-1)
    y_preds.append(test_y_pred)
    y_.append(test_y)
    resroot = "/Users/lizhenhao/Desktop/helloworld/im/resnet50"
    res = fc.get_model(2048)
    res.load_weights("/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/resnet50/epoch-19.h5")
    test_f, test_y = getdata_fromtxt(resroot + '/test.txt')
    test_y_pred = res.predict(test_f)
    test_y_pred = test_y_pred.reshape(-1)
    y_preds.append(test_y_pred)
    y_.append(test_y)

    draw.draw_roc(y_, y_preds, name_list, 'roc.png')
if __name__=='__main__':
    # model=vggtrain()
    # root = '/Users/lizhenhao/Desktop/helloworld/im/vgg'
    # model.load_weights("/Users/lizhenhao/PycharmProjects/laserFilm/weights/freeze/vgg/epoch--19.h5")
    # test_f, test_y = getdata_fromtxt(root + '/test.txt')
    # test_y_pred = model.predict(test_f)
    # test_y_pred = test_y_pred.reshape(-1)
    # y_preds = []
    # name_list = ['vgg']
    # y_preds.append(test_y_pred)
    # draw.draw_roc(test_y, y_preds, name_list,'roc.png')
    # vgg=vggtrain()
    # iv3=iv3train()
    # densenet=densenettrain()
    # resnet=resnet50train()
    #plot_roc()
    vggtrain()


