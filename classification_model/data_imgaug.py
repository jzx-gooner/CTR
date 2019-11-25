#coding=utf-8
#数据集扩充
# ./images/ data1 data2 data3
# python data_imgaug.py
#参考:https://github.com/aleju/imgaug
from imgaug import augmenters as iaa
from PIL import Image
from numpy import *
import numpy as np
import os
import os.path
from shutil import copyfile
from math import floor
from random import shuffle
import matplotlib.pylab as plt
import time, threading

nums=int(raw_input("需要将数据集扩充到多少张："))
src_dir = './dataset'

def thread_work(sblst_sub):
    imgs = []
    imgpath = []
    for pic_name in sblst_sub:
        try:
            filepath_src = src_dir + '/' + lst[ind] + '/' + pic_name
            if pic_name.endswith('.png') == True:
                continue
            else:
                img = plt.imread(filepath_src)
                imgs.append(img)
                imgpath.append(src_dir + '/' + lst[ind] + '/' + 'aug' + pic_name)
        except(IOError), e:
            print e
            continue

        else:
            continue

    images_aug = seq.augment_images(imgs)
    for i in range(0, len(imgs)):
        try:
            #plt.imshow(images_aug[i])
            plt.imsave(imgpath[i], images_aug[i])
        except(ValueError),e:
            print e
            continue
        else:
            continue

seq = iaa.OneOf([
    iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Flipud(1), #Flip vertically 
    iaa.Fliplr(1), # horizontally flip
    iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
    iaa.ContrastNormalization((0.75, 1.5)),  # Strengthen or weaken the contrast in each image.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # Add gaussian noise.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),  # Make some images brighter and some darker.
    iaa.AdditiveGaussianNoise(scale=0.1*255),
    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
    #iaa.Affine(translate_px={"x": -40}),  # Augmenter to apply affine transformations to images.
    #iaa.Scale({"height": 512, "width": 512}),
    iaa.WithChannels(0, iaa.Add((10, 100))),
    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
    iaa.Add((-40, 40)),
    #iaa.Multiply((0.5, 1.5), per_channel=0.5),
    iaa.CoarseDropout(0.02, size_percent=0.5),
    iaa.ContrastNormalization((0.5, 1.5)),
    #weather
    #iaa.FastSnowyLandscape(lightness_threshold=[100, 200], lightness_multiplier=[1.0, 4.0])

	
])


class_names = []
for filename in os.listdir(src_dir):
    path = os.path.join(src_dir, filename)
    if os.path.isdir(path):
        class_names.append(filename)

lst = class_names
thread_num = 24;
for ind in range(0,len(lst)):
    sblst=os.listdir(os.path.join(src_dir,lst[ind]))
    shuffle(sblst)
    print("processing "+lst[ind]+"...")
    print(len(sblst))

    # skip the dir already big enough
    if len(sblst)>nums:
        print "skip this data!"
        continue

    threads_pool = []

    for i in range(thread_num-1):
        # multi thread here
        batch_size = len(sblst)/thread_num
        t = threading.Thread(target=thread_work, args=(sblst[i*batch_size:(i+1)*batch_size],))
        threads_pool.append(t);
        t.start()

    # specially process the last thread
    t = threading.Thread(target=thread_work, args=(sblst[(i+1)*batch_size:],))
    threads_pool.append(t)
    t.start()

    # join the threads
    for i in range(0,len(threads_pool)):
        threads_pool[i].join()

print "done!"
