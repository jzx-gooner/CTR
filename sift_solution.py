# coding=utf-8
# Author:jzx
# Creation Date:2019/9/3
# sift + cnn classification

import time
import argparse
from os.path import join
from os import walk
import numpy as np
import cv2
import xlrd
import xlsxwriter
import shutil
import sys

reload(sys)
sys.setdefaultencoding('utf8')
if sys.getdefaultencoding() != 'gbk':
    reload(sys)
    sys.setdefaultencoding('gbk')

class SiftSolution(object):
    def __init__(self):
        self.W_DEFAULT = 1024
        self.H_DEFAULT = 768
        self.MIN_MATCH_COUNT = 70    
    def crop_img(self,source_img, query_img_path):
        source_img = cv2.imread(source_img, 0)
        query_img = cv2.imread(query_img_path)
        query_img = cv2.resize(query_img, (self.W_DEFAULT, self.H_DEFAULT))
        img_raw_copy = query_img.copy()
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        # 计算关键点和描述符
        kp1, des1 = sift.detectAndCompute(source_img, None)
        kp2, des2 = sift.detectAndCompute(query_img, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, 2)
        # 寻找距离近的放入good列表
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # 原始图像关键点
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # 训练图像关键点
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # 计算单应性
            matchesMask = mask.ravel().tolist()  # 转成一维度,然后转成量表
            # 从good中选出更good的
            selected = [v for k, v in enumerate(good) if matchesMask[k]]
            # 针对所有的selected点再次计算出更精确的转化矩阵M来
            sch_pts, dst_pts = np.float32([kp1[m.queryIdx].pt for m in selected]).reshape(
                -1, 1, 2), np.float32([kp2[m.trainIdx].pt for m in selected]).reshape(-1, 1, 2)
            M2, mask = cv2.findHomography(sch_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = source_img.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst2 = cv2.perspectiveTransform(pts, M2)
            query_img = cv2.polylines(img_raw_copy, [np.int32(dst2)], True, (0, 255, 0), 3, cv2.LINE_AA)
            draw_params = dict(matchColor=(0, 255, 0),
                            singlePointColor=None,
                            matchesMask=matchesMask,
                            flags=2
                            )
            img3 = cv2.drawMatches(source_img, kp1, query_img, kp2, good, None, **draw_params)
            cv2.putText(img3, str(len(good)), (50, 150), cv2.FONT_HERSHEY_COMPLEX, 6, (0, 0, 255), 25)
            # 裁剪变换视角
            perspectvieM = cv2.getPerspectiveTransform(np.float32(dst2), pts)
            found = cv2.warpPerspective(img_raw_copy, perspectvieM, (w, h))
            temp_name = query_img_path.split("/")[-1]
            cv2.imwrite("./debug/temp/" + temp_name, found)
            cv2.imwrite("./debug/temp_match/" + temp_name, img3)
        else:
            shutil.copy(query_img_path, "./debug/temp/")

    def find_match_index(self,source, query_img_path):
        query = cv2.imread(query_img_path, 0)
        query = cv2.resize(query, (self.W_DEFAULT, self.H_DEFAULT))
        files = []
        images = []
        descriptors = []
        for (dirpath, dirnames, filename) in walk(source):
            files.extend(filename)
            for f in files:
                if f.endswith("npy"):
                    descriptors.append(f)
            print(descriptors)

        # create the sift detector
        sift = cv2.xfeatures2d.SIFT_create()
        query_kp, query_ds = sift.detectAndCompute(query, None)

        # create FLANN matcher
        FLANN_INDEC_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEC_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        potential_culprits = {}
        print(">>>>>Iintiating picture scan")
        for d in descriptors:
            # print("--------analyzing {} for matches -----------".format(d))
            matches = flann.knnMatch(query_ds, np.load(join(source, d)), k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) > self.MIN_MATCH_COUNT:
                print("{} is a match ! {} ".format(d, len(good)))
                is_pass_min_match = True
            else:
                print("{} is not a match".format(d))
                is_pass_min_match = False
            potential_culprits[d] = len(good)
        max_matches = None
        potential_suspect = None
        for culprit, matches in potential_culprits.iteritems():
            if max_matches == None or matches > max_matches:
                max_matches = matches
                potental_suspect = culprit
        print("potential suspect is {}".format(potental_suspect.replace("npy", "").upper()))
        # reload the best
        best_match_img_name = source + potental_suspect.replace(".npy", "")
        return best_match_img_name, is_pass_min_match


def read_excel(file):
    data = xlrd.open_workbook(filename=file)  # 打开文件#此时data相当于指向该文件的指针
    table = data.sheet_by_index(0)  # 通过索引获取表格
    names = data.sheet_names()
    print(names[0].encode('utf-8'))
    row = int(raw_input("请输入图片所在的行: "))
    column = int(raw_input("请输入图片所在的列: "))
    col_images = table.col_values(column)
    img_list = []  # 图片索引
    for item_col_images in col_images:
        name = item_col_images.encode("utf-8")
        path = "./images/" + name + ".jpg"
        img_list.append(path)
    # print row_3.index("实际广告内容").encode('utf-8') #获得索引
    print(img_list[row:])
    return img_list[row:]


def write_excel(results, excel_path):
    workbook = xlsxwriter.Workbook(excel_path)
    worksheet = workbook.add_worksheet()
    worksheet.write_column('U4', excel_path)
    workbook.close()

    #print(finetune_model.summary())

if __name__ == '__main__':
    excel_path = "./框架数据明细.xlsx"
    source = "./source/"
    img_list = read_excel(excel_path)
    print(len(img_list))
    print(len(set(img_list)))
    # assert len(img_list)==len(set(img_list))
    solution1_results= []
    sift_soulution = SiftSolution()
    for query in img_list:
        # solution 1: sift
        best_match_img_name, is_pass_min_match = sift_soulution.find_match_index(source, query)
        if is_pass_min_match:
            sift_soulution.crop_img(best_match_img_name + '.jpg', query)
            solution1_results.append([query,best_match_img_name.split("/")[-1]])
        else:
            solution1_results.append([query,"UNCERTAIN"])
            shutil.copy(query, "./debug/temp/")
