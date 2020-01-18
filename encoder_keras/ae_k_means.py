#coding=utf-8
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['font.size'] = 8
plt.rcParams['figure.figsize'] = (8,8)

import os
import numpy as np
import cv2
import pickle

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K

from sklearn.cluster import KMeans

img_width, img_height, channels = 512, 512, 3
input_shape = (img_width, img_height, channels)

def load_data():
    dir = 'images'
    files = ["%s/%s" % (dir, x) for x in os.listdir(dir)]
    arr = np.empty((len(files), img_width, img_height, channels), dtype=np.float32)
    for i, imgfile in enumerate(files):
        img = cv2.imread(imgfile)
        x = cv2.resize(img,(512,512))
        x = x.astype('float32') / 255.
        arr[i] = x
    return arr

X = load_data()

print(X.shape)
nb_rows, nb_cols = 3, 3
plt.figure(figsize=(3,3))

for k in range(nb_rows * nb_cols):
    plt.subplot(nb_rows, nb_cols, k+1)
    plt.imshow(X[k])
    plt.axis('off')
#plt.show()

#np.save('test.npy', X)

# 搭建 autoencoder 模型
model = Sequential()
# encoder 部分: ( conv + relu + maxpooling ) * 3
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# decoder 部分: ( conv + relu + upsampling ) * 3 与 encoder 过程相反
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()

# 训练
#model.fit(X, X, epochs=100, batch_size=64, shuffle=True)

# 检查 autoencoder 的效果
def plot_some(im_list):
    plt.figure(figsize=(15,4))
    for i, array in enumerate(im_list):
        plt.subplot(1, len(im_list), i+1)
        plt.imshow(array)
        plt.axis('off')
    #plt.show()

img_decoded = model.predict(X[:5])
print('Before autoencoding:')
plot_some(X[:5])
print('After decoding:')
plot_some(img_decoded)
plt.show()

model.save('encodermodel.h5')


get_encoded = K.function([model.layers[0].input], [model.layers[5].output])
# 取5个样本分析
X_sample = X[:5]
print(X_sample.shape)

# 创建一个获取图片 encoded 结果的函数, 即取模型的 encoder 部分 (前六层)
# Keras 如何获取中间层, 参考官方文档:
# https://keras.io/zh/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
# 获取样本的 encoded 结果
encoded_sample = get_encoded([X_sample])[0]

print(encoded_sample.shape)

# 看下 encode 之后的图片是什么情况
# 因为 shape 是 16*16*8, 因此需要转成灰度才能将图片显示出来, 分别将最后的维度用 mean, max, std 来取值
# 从结果来看, 部分图片隐约还能看到个轮廓

for n_image in range(0, 5):
    
    plt.figure(figsize=(12,4))

    plt.subplot(1,4,1)
    plt.imshow(X_sample[n_image][:,:,::-1])
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1,4,2)
    plt.imshow(encoded_sample[n_image].mean(axis=-1))
    plt.axis('off')
    plt.title('Encoded Mean')

    plt.subplot(1,4,3)
    plt.imshow(encoded_sample[n_image].max(axis=-1))
    plt.axis('off')
    plt.title('Encoded Max')

    plt.subplot(1,4,4)
    plt.imshow(encoded_sample[n_image].std(axis=-1))
    plt.axis('off')
    plt.title('Encoded Std')

    plt.show()

# 所有图片 encoded 之后的数据
X_encoded = np.empty((len(X), 64, 64, 8), dtype='float32')
step = 100
for i in range(0, len(X), step):
    x_batch = get_encoded([X[i:i+step]])[0]
    X_encoded[i:i+step] = x_batch

print(X_encoded.shape)
# reshape, 其实相当于 flatten, 之后给 KMeans 用
X_encoded_reshape = X_encoded.reshape(X_encoded.shape[0], X_encoded.shape[1]*X_encoded.shape[2]*X_encoded.shape[3])
print(X_encoded_reshape.shape)
#聚类
n_clusters = 15
km = KMeans(n_clusters=n_clusters)
km.fit(X_encoded_reshape)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=100, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

# 看下 clusters 的分布结果
plt.figure(figsize=(20,5))
cluster_elements = [(km.labels_==i).sum() for i in range(n_clusters)]
plt.bar(range(n_clusters), cluster_elements, width=1)
#plt.show()
# 每个聚类的 encoded 均值

average_clusters_encoded = []
for i in range(n_clusters):
    average_clusters_encoded.append(X_encoded[km.labels_==i].mean(axis=0))

average_clusters_encoded = np.asarray(average_clusters_encoded)
print(average_clusters_encoded.shape)
# 取出模型的 decoder 部分

get_decoded = K.function([model.layers[6].input],
                         [model.layers[-1].output])

# "平均" decoded 图像
decoded_clusters = get_decoded([average_clusters_encoded])

# 打印出所有聚类的" 平均" decoded 图像

plt.figure(figsize=(20,20))

for i in range(n_clusters):
    plt.subplot(10, 10, i+1)
    plt.imshow(decoded_clusters[0][i][:,:,::-1])
    plt.title('Cluster {}'.format(i))
    plt.axis('off')

plt.show()

#显示部分图像


plt.figure(figsize=(20, 20))

cluster = 0
rows, cols = 2, 2
start = 0

labels = np.where(km.labels_==cluster)[0][start:]
for i, label in enumerate(labels):
    print(labels)
    plt.subplot(rows, cols, i+1)
    plt.imshow(X[label])
    plt.axis('off')
plt.show()
