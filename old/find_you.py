# coding=utf-8
# Author:jzx
# Creation Date:2019/9/3
# sift + cnn classification
from keras.applications.mobilenetv2 import MobileNetV2
from keras import backend as K
from keras.applications.mobilenet import preprocess_input
from classification_model import utils
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

USE_CAM_FOR_DEBUG = True
NUM_CLASS = 2
MIN_MATCH_COUNT = 70
W_DEFAULT = 1024
H_DEFAULT = 768


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


def crop_img(source_img, query_img_path):
    source_img = cv2.imread(source_img, 0)
    query_img = cv2.imread(query_img_path)
    query_img = cv2.resize(query_img, (W_DEFAULT, H_DEFAULT))
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
    if len(good) > MIN_MATCH_COUNT:
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

def find_match_index(source, query_img_path):
    query = cv2.imread(query_img_path, 0)
    query = cv2.resize(query, (W_DEFAULT, H_DEFAULT))
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
        if len(good) > MIN_MATCH_COUNT:
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

if USE_CAM_FOR_DEBUG:
    parser = argparse.ArgumentParser()
    parser.add_argument('--resize_height', type=int, default=128, help='Height of cropped input image to network')
    parser.add_argument('--resize_width', type=int, default=128, help='Width of cropped input image to network')
    parser.add_argument('--alpha', type=float, default="1", help='the aplha of the model')
    args = parser.parse_args()
    WIDTH = args.resize_width
    HEIGHT = args.resize_height
    ALPHA = args.alpha
    preprocessing_function = preprocess_input
    base_model = MobileNetV2(weights='imagenet',alpha=ALPHA, include_top=False, input_shape=(WIDTH, HEIGHT, 3))
    FC_LAYERS=[1024]
    #load model
    class_list_file = "./classification_model/checkpoints/" + "MobileNetV2" + "_" + "dataset" + "_class_list.txt"
    class_list = utils.load_class_list(class_list_file)
    finetune_model = utils.build_finetune_model(base_model, dropout=1e-3, fc_layers=FC_LAYERS, num_classes=NUM_CLASS)
    finetune_model.load_weights("./classification_model/checkpoints/" + "MobileNetV2" + "_model_weights.h5")
    #print(finetune_model.summary())

if __name__ == '__main__':
    excel_path = "./框架数据明细.xlsx"
    source = "./source/"
    img_list = read_excel(excel_path)
    print(len(img_list))
    print(len(set(img_list)))
    # assert len(img_list)==len(set(img_list))
    solution1_results = []
    solution2_results = []
    for query in img_list:
        # solution 1: sift
        best_match_img_name, is_pass_min_match = find_match_index(source, query)
        if is_pass_min_match:
            crop_img(best_match_img_name + '.jpg', query)
            solution1_results.append([query,best_match_img_name.split("/")[-1]])
        else:
            solution1_results.append([query,"UNCERTAIN"])
            shutil.copy(query, "./debug/temp/")

        # solution 2: cnn
        if USE_CAM_FOR_DEBUG:
            image = cv2.imread(query,-1)
            image = np.float32(cv2.resize(image, (WIDTH, HEIGHT)))
            image = preprocessing_function(image.reshape(1, WIDTH, HEIGHT, 3))
            # Finally we preprocess the batch
            # (this does channel-wise color normalization)
            out = finetune_model.predict(image)
            print(out)
            class_prediction = list(out[0]).index(max(out[0]))
            class_name = class_list[class_prediction]
            if max(out[0])<0.7:
                solution2_results.append([query,"UNCERTAIN"])
            else:
                solution2_results.append([query,class_name[0]])
            blade_tip_output = finetune_model.output[:,int(class_prediction)]
            last_conv_layer = finetune_model.get_layer('Conv_1_bn')
            grads = K.gradients(blade_tip_output, last_conv_layer.output)[0]
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
            iterate = K.function([finetune_model.input], [pooled_grads, last_conv_layer.output[0]])
            pooled_grads_value, conv_layer_output_value = iterate([image])
            for i in range(1280):
                conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

            heatmap = np.mean(conv_layer_output_value, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            img = cv2.imread(query)
            img = cv2.resize(img, (WIDTH, HEIGHT))
            # # We resize the heatmap to have the same size as the original image
            heatmap = cv2.resize(heatmap, (WIDTH, HEIGHT))
            # # We convert the heatmap to RGB
            heatmap = np.uint8(255 * heatmap)
            # # We apply the heatmap to the original image
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # # 0.4 here is a heatmap intensity factor
            superimposed_img = heatmap * 0.4 + img
            # # Save the image to disk
            now=time.time()
            #imgs = np.hstack([img,superimposed_img])
            cv2.putText(superimposed_img,class_name[0] , (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite('./debug/temp_cam/'+str(class_name[0])+"_"+query.split("/")[-1].split(".")[0]+'.jpg', superimposed_img)


    print(solution1_results)
    #print(solution2_results)



    # write_excel(results,excel_path)
