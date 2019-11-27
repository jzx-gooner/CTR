#coding=utf-8
#分类模型，进行预测，生成cam图，debug
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K
import cv2
from keras.applications.mobilenet import preprocess_input
import utils
import glob
import time
import argparse

NUM_CLASS = 2

parser = argparse.ArgumentParser()
parser.add_argument('--resize_height', type=int, default=128, help='Height of cropped input image to network')
parser.add_argument('--resize_width', type=int, default=128, help='Width of cropped input image to network')
parser.add_argument('--alpha', type=float, default="0.35", help='the aplha of the model')
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
print(finetune_model.summary())


# The local path to our target image
for img_path in glob.glob('./test/*'):
    image = cv2.imread(img_path,-1)
    image = np.float32(cv2.resize(image, (WIDTH, HEIGHT)))
    image = preprocessing_function(image.reshape(1, WIDTH, HEIGHT, 3))


    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    out = finetune_model.predict(image)
    class_prediction = list(out[0]).index(max(out[0]))
    class_name = class_list[class_prediction]
    blade_tip_output = finetune_model.output[:,int(class_prediction)]

    # # The is the output feature map of the `block5_conv3` layer,
    # # the last convolutional layer in VGG16
    last_conv_layer = finetune_model.get_layer('Conv_1_bn')

    # # This is the gradient of the "african elephant" class with regard to
    # # the output feature map of `block5_conv3`
    grads = K.gradients(blade_tip_output, last_conv_layer.output)[0]

    # # This is a vector of shape (512,), where each entry
    # # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # # This function allows us to access the values of the quantities we just defined:
    # # `pooled_grads` and the output feature map of `block5_conv3`,
    # # given a sample image
    iterate = K.function([finetune_model.input], [pooled_grads, last_conv_layer.output[0]])

    # # These are the values of these two quantities, as Numpy arrays,
    # # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([image])
    # # We multiply each channel in the feature map array
    # # by "how important this channel is" with regard to the elephant class
    for i in range(1280):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # # The channel-wise mean of the resulting feature map
    # # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #plt.matshow(heatmap)
    #plt.show()


    # # We use cv2 to load the original image
    img = cv2.imread(img_path)
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
    imgs = np.hstack([img,superimposed_img])
    cv2.imwrite('./heatmap/'+str(class_name[0])+str(now)+'.jpg', imgs)
