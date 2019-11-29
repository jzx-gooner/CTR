# coding=utf-8
# Author:jzx
# Creation Date:2019/9/3
# sift + cnn classification
from keras.applications.mobilenetv2 import MobileNetV2
from keras import backend as K
from keras.applications.mobilenet import preprocess_input
from classification_model import utils
import numpy as np
from keras import backend as K
import cv2
import glob
import time
from excel_tools import ExcelTool
import sys

reload(sys)
sys.setdefaultencoding('utf8')
if sys.getdefaultencoding() != 'gbk':
    reload(sys)
    sys.setdefaultencoding('gbk')
class CnnSolution(object):
    def __init__(self):
        self.USE_CAM_FOR_DEBUG = True 
        self.NUM_CLASS = 2
        self.WIDTH = 128
        self.HEIGHT = 128
        self.ALPHA = 1.0
        self.preprocessing_function = preprocess_input
        base_model = MobileNetV2(weights='imagenet',alpha=self.ALPHA, include_top=False, input_shape=(self.WIDTH, self.HEIGHT, 3))
        self.FC_LAYERS=[1024]
        #load model
        self.class_list_file = "./classification_model/checkpoints/" + "MobileNetV2" + "_" + "dataset" + "_class_list.txt"
	self.weights_path = "./classification_model/checkpoints/" + "MobileNetV2" + "_model_weights.h5"
	print(self.class_list_file)
        self.class_list = utils.load_class_list(self.class_list_file)
        self.finetune_model = utils.build_finetune_model(base_model, dropout=1e-3, fc_layers=self.FC_LAYERS, num_classes=self.NUM_CLASS)
        self.finetune_model.load_weights(self.weights_path)
        self.result = "NONE"
        print(self.finetune_model.summary())

    def cnn_predict(self,query):
        image = cv2.imread(query,-1)
        image = np.float32(cv2.resize(image, (self.WIDTH, self.HEIGHT)))
        image = self.preprocessing_function(image.reshape(1, self.WIDTH, self.HEIGHT, 3))
        # Finally we preprocess the batch
        # (this does channel-wise color normalization)
        out = self.finetune_model.predict(image)
        print(out)
        class_prediction = list(out[0]).index(max(out[0]))
        class_name = self.class_list[class_prediction]
        if max(out[0])<0.7:
            self.result = "UNCERTAIN"
        else:
            self.result = class_name[0]
        blade_tip_output = self.finetune_model.output[:,int(class_prediction)]
        last_conv_layer = self.finetune_model.get_layer('Conv_1_bn')
        grads = K.gradients(blade_tip_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([self.finetune_model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([image])
        for i in range(1280):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        img = cv2.imread(query)
        img = cv2.resize(img, (self.WIDTH, self.HEIGHT))
        # # We resize the heatmap to have the same size as the original image
        heatmap = cv2.resize(heatmap, (self.WIDTH, self.HEIGHT))
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
        return self.result


if __name__ == '__main__':
    excel_path = "./框架数据明细.xlsx"
    source = "./source/"
    excle_tool = ExcelTool()
    img_list = excle_tool.read_excel(excel_path)
    cnn_predict = CnnSolution()
    # assert len(img_list)==len(set(img_list))
    for query in img_list:
        result = cnn_predict.cnn_predict(query)
        print (result)
