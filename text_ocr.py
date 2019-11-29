# -*- coding:utf-8 -*-

import sys
import os

sys.path.append(os.getcwd() + "/chinese_ocr")
sys.path.append(os.getcwd() + "/chinese_ocr/ctpn")
print(sys.path)
import chinese_ocr.ocr as ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
from stupid_correct import find_match_words

image_files = glob('./debug/temp/*.*')

if __name__ == '__main__':
    result_dir = './debug/temp_ocr'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    ocr_result_list = []
    for image_file in sorted(image_files):
        im = Image.open(image_file).convert('RGB')
        width, height = im.size
        new_size = (width, int(height * 1.1))
        im = im.resize(new_size)
        image = np.array(im)
        t = time.time()
        result, image_framed ,fx,fy = ocr.model(image)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        if len(result) == 0:
            print("完全没识别出字来")
            ocr_result_list.append("UNCERTAIN")
        else:
            for key in result:
                print result[key][1]
                ocr_flag, ocr_result = find_match_words(result[key][1])
                if ocr_flag:
                    #print(result[key][0])
                    ocr_result_list.append(ocr_result)
                    print("识别出结果")
                else:
                    ocr_result_list.append("UNCERTAIN")
        print(ocr_result_list)