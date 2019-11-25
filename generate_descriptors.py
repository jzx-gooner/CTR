#coding=utf-8
#Author:jzx
#Creation Date:2019/9/3
import cv2
import numpy as np
from os import walk
from os.path import join
import sys

def save_descriptor(folder,image_path,feature_detector):
    img=cv2.imread(join(folder,image_path),0)
    img = cv2.resize(img,(224,224))
    keypoints,descriptors=feature_detector.detectAndCompute(img,None)
    descriptor_file=image_path.replace('jpg' or 'JPG',"npy")
    np.save(join(folder,descriptor_file),descriptors)

def create_descriptors(folder):
    files=[]
    for (dirpath,dirnames,filenames) in walk(folder):
        files.extend(filenames)
    for f in files:
        save_descriptor(folder,f,cv2.xfeatures2d.SIFT_create())

if __name__ == '__main__':
    dir="./source"
    create_descriptors(dir)
