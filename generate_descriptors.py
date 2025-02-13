#coding=utf-8
#Author:jzx
#Creation Date:2019/9/3
import cv2
import numpy as np
from os import walk
from os.path import join

#一般认为图片占比为1/2
W_DEFAULT = int(1024*0.5)
H_DEFAULT = 768


def save_descriptor(folder,image_path,feature_detector):
    img=cv2.imread(join(folder,image_path),0)
    img = cv2.resize(img,(W_DEFAULT,H_DEFAULT))
    keypoints,descriptors=feature_detector.detectAndCompute(img,None)
    print(image_path)
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
