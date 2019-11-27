# -*- coding:utf-8 -*-

import sys
import os

sys.path.append(os.getcwd() + "/chinese_ocr")
sys.path.append(os.getcwd() + "/chinese_ocr/ctpn")
print(sys.path)
import os
import chinese_ocr.ocr as ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
import cv2

image_files = glob('./temp/*.*')

if __name__ == '__main__':
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        im = Image.open(image_file).convert('RGB')
        width, height = im.size
        new_size = (width, int(height * 1.1))
        im = im.resize(new_size)
        image = np.array(im)
        t = time.time()
        result, image_framed = ocr.model(image)
        print(result)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][0])
            print(result[key][1])
            if result[key][1] =="衡水威宝0318515666":
                print (result[key][1])
