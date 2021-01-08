
# /Users/lizhenhao/Desktop/hello\ world/毕业设计/镭雕显微图片/data
import os
import random
import cv2
import numpy as np
from pathlib import Path


def delete(path):
    f=Path(path)
    if f.is_file():
        os.remove(path)
    else:
        flag=True
        for (root, dirs, files) in os.walk(path):
            if flag:
                curRoot=root
                flag=False
                for x in files:
                    try:
                        os.remove(os.path.join(root,x))
                    except Exception as e:
                        print('Exception:', e)
            for item in dirs:
               delete(os.path.join(root,item))
        try:
            os.rmdir(curRoot)
        except Exception as e:
            print('Exception:', e)
def move(src,dst):
    image=cv2.imread(src)
    cv2.imwrite(dst,image)
# def solve ():
#     dir = "/Users/lizhenhao/Desktop/helloworld/毕业设计/镭雕显微图片/data"
#     test_good = 30
#     test_bad = 30
#     val_good = 30
#     val_bad = 30
#     # 正样本转移60张到test
#     # 负样本转移60张到val
#     for rootg, _, goods in os.walk(dir + "/good"):
#         print("read good over")
#     for rootb, _, bads in os.walk(dir + "/bad"):
#         print("read bad over")
#     # 正样本随机数
#     tol_g = len(goods)
#     rg = random.sample(range(0, tol_g), test_good + val_good)
#     p=0
#     tar="/train/"
#     index=0
#     for j in goods:
#         if (index in rg):
#             move(rootg+"/"+j,dir+tar+"g"+j)
#             p=p+1
#         if(p==test_good):
#             tar="/val/"
#         index+=1
#     # 负样本随机数
#     tol_b = len(bads)
#     p=0
#     index=0
#     tar="/train/"
#     bg = random.sample(range(0, tol_b), test_good + val_good)
#     for j in bads:
#         if (index in bg):
#             move(rootb + "/" + j, dir + tar + "b" + j)
#             p=p+1
#         if (p == test_good):
#             tar = "/val/"
#         index+=1


def print_dataInfo(root):
    begin=True
    for __, subs, fs in os.walk(root):
        if begin:
            print(__+" contains %d subdir"%(len(subs)))
            begin=False
        else:
            print("    " + __ + " include %d images" % (len(fs)))


def split_data(dataPath,targetPath,splitRatio):
    """
    :param dataPath: 源文件夹路径。此路径下包含若干子文件夹，每个子文件夹代表一个待分类类别，里面包含该类型的所有图片。
    :param targetPath: 数据集生成路径。路径下会生成三个包分别为训练集、验证集、测试集。每个集合下会生成若干个类别。
    :param splitRatio: 按比例划分数据集。训练集：验证集  或  训练集：验证集：测试集
    :return: 空
    """
    if not os.path.exists(dataPath):
        raise Exception("数据文件夹内没有数据，无法执行")
    if len(splitRatio)==0:
        raise Exception("划分比例数组不能为空")
    flag=True
    for root,subs,k in os.walk(dataPath):
        if flag:
            break;
    noc=len(subs)
    dataNum=[0]*noc
    for i in range(noc):
        for _,__,f in os.walk(dataPath+"/"+subs[i]):
            dataNum[i]=len(f)
    print_dataInfo(dataPath)
    if len(splitRatio)==2:
        setName=["train","vali"]
    else:
        setName=["train","vali","test"]
    for i in range(len(splitRatio)):
        if os.path.exists(targetPath+"/"+setName[i]):
            try:
                delete(targetPath+"/"+setName[i])
            except Exception as e:
                print('Exception:',e)
        os.mkdir(targetPath+"/"+setName[i])
        [os.mkdir(x) for x in [targetPath+"/"+setName[i]+"/"+cla for cla in subs]]
    setNum=np.array(splitRatio/np.sum(splitRatio)*np.sum(dataNum)).astype(int)
    print("processing>>>>>>>>>")
    for i in range(len(subs)):
        for _,__,images in os.walk(dataPath+"/"+subs[i]):
            shuffleIndex=np.random.permutation(np.arange(len(images)))
            curIndex=0
            for j in range(len(setNum)):
                adder=int(setNum[j]*dataNum[i]/np.sum(dataNum))
                p=0
                for k in range(adder):
                    move(dataPath+"/"+subs[i]+"/"+images[shuffleIndex[curIndex+k]],targetPath+"/"+setName[j]+"/"+subs[i]+"/"+str(p)+".jpg")
                    p+=1
                curIndex += adder
    for set in setName:
        print_dataInfo(targetPath+"/"+set)
if __name__ == '__main__':
    root="/Users/lizhenhao/Desktop/helloworld/毕业设计/镭雕显微图片/origion"
    tar="/Users/lizhenhao/Desktop/helloworld/im"
    split_data(root,tar,[8,1,1])
