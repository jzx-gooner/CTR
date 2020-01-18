#coding=utf-8
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

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

def load_data():
 
    arr = np.empty((1, 512, 512, 3), dtype=np.float32)
    img = cv2.imread("./1.jpg")
    x = cv2.resize(img,(512,512))
    x = x.astype('float32') / 255.
    arr = x
    return arr


X = load_data()

model=load_model("./encodermodel.h5")
print(model.summary())

get_encoded = K.function([model.layers[0].input], [model.layers[5].output])
X_sample = X[:]
encoded_sample = get_encoded([X_sample])[0]

# # 所有图片 encoded 之后的数据
X_encoded = np.empty((len(X), 64, 64, 8), dtype='float32')
step = 100
for i in range(0, len(X), step):
    x_batch = get_encoded([X[i:i+step]])[0]
    X_encoded[i:i+step] = x_batch

X_encoded_reshape = X_encoded.reshape(X_encoded.shape[0], X_encoded.shape[1]*X_encoded.shape[2]*X_encoded.shape[3])
print(X_encoded_reshape.shape)
#聚类
n_clusters = 3
km = KMeans(n_clusters=n_clusters)
km.fit(X_encoded_reshape)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=100, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
#
cluster = 0
labels = np.where(km.labels_==cluster)[0][0:]
print("1111111111111")
print(labels)
for i, label in enumerate(labels):
    plt.imshow(X[label])
    plt.show()