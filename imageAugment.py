import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


import glob



# 设置生成器参数
def solve():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        channel_shift_range=10
    )
    dir="/Users/lizhenhao/Desktop/helloworld/毕业设计/镭雕显微图片/origion/cx"
    save_dir="/Users/lizhenhao/Desktop/helloworld/pic"
    gen_data = datagen.flow_from_directory(dir,
                                           batch_size=1,
                                           shuffle=False,
                                           save_to_dir=save_dir,
                                           save_prefix='gen')
    for i in range(1000):
        gen_data.next()
if __name__ == '__main__':
    solve()
    print("done")