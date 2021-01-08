import tensorflow as tf
import dataSplit as ds
from tensorflow.keras.utils import plot_model
import numpy as np
import cv2
import os
import util.txtutil as tu



def forward (model, dataPath, savePath,model_name):
    feature_dir = os.path.join(savePath, model_name)
    set_name = ['train', 'test', 'vali']
    if os.path.exists(feature_dir):
        ds.delete(feature_dir)
    os.mkdir(feature_dir)
    # 每个训练包单独处理 train test vali
    for set in set_name:
        file_name = feature_dir+"/"+set + ".txt"
        txtName=open(file_name,'w')
        set_dir = os.path.join(dataPath, set)
        first = True
        label = [-1, 1]
        p = 0
        for root, sub, file in os.walk(set_dir):
            if first:
                first = False
                continue
            noi=0
            input = []
            # 读如图片整合为向量
            for image in file:
                image_path = os.path.join(root, image)
                if image_path.split(".")[-1]!="jpg":
                    continue
                print(image_path)
                im = cv2.imread(image_path)
                im = cv2.resize(im, (224, 224))
                im = im / 255
                input.append(im)
                noi+=1
            la = [label[p]] * noi
            p += 1
            print("输入网络")
            fea = tf.keras.Model(inputs=model.inputs, outputs=model.output).predict(
                np.array(input))
            print("向量整合")
            fea=fea.reshape(-1,2048)
            features = np.array(fea)
            labels = np.array(la)
            labels = labels[:, np.newaxis]
            ans = np.hstack((features, labels))
            print("写入txt中")
            tu.saveTxt(txtName, ans)
        txtName.close()




# vgg:block5_pool:25088
# densenet:relu:50176
# resnet: model.output:2048
# iv3:mixed10:51200
if __name__ == '__main__':
    dataPath="/Users/lizhenhao/Desktop/helloworld/im"
    savePath="/Users/lizhenhao/Desktop/helloworld/im"
    # vgg
    #vgg=tf.keras.applications.vgg19.VGG19(weights="imagenet",include_top=False)
    #forward(vgg,dataPath,savePath,"vgg")
    # resnet
    # resnet50=tf.keras.applications.resnet50.ResNet50(weights='imagenet',include_top=False)
    # x=resnet50.output
    # x=tf.keras.layers.GlobalAveragePooling2D()(x)
    # myresnet=tf.keras.models.Model(inputs=resnet50.inputs,outputs=x)
    # forward(myresnet,dataPath,savePath,"resnet50")
    #inception_v3
    #iv3=tf.keras.applications.inception_v3.InceptionV3(weights='imagenet',include_top=False)
    #forward(iv3, dataPath, savePath, "iv3")
    # densenet
    #den121=tf.keras.applications.densenet.DenseNet121(weights='imagenet',include_top=False)
    #forward(den121, dataPath, savePath, "densenet")
    print("done")