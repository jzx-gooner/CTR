#coding=utf-8
#blade classify
from __future__ import print_function

# Networks
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from SqueezeNet import SqueezeNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.preprocessing.image import ImageDataGenerator

# Layers
from keras.layers import Dense, Activation, Flatten, Dropout
from keras import backend as K

# Other
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping,TensorBoard
from keras.models import load_model

# Utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random, glob
import os, sys, csv
import cv2
import time, datetime

# Files
import utils
# For boolean input from the command line
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Command line argspretrained_weights
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=5000, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train", help='Select "train", or "predict" mode. \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="dataset", help='Dataset you are using.')
parser.add_argument('--resize_height', type=int, default=128, help='Height of cropped input image to network')
parser.add_argument('--resize_width', type=int, default=128, help='Width of cropped input image to network')
parser.add_argument('--alpha', type=float, default="0.35", help='the aplha of the model')
parser.add_argument('--batch_size', type=int, default=8, help='Number of images in each batch')
parser.add_argument('--dropout', type=float, default=1e-3, help='Dropout ratio')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--rotation', type=float, default=90.0, help='Whether to randomly rotate the image for data augmentation')
parser.add_argument('--zoom', type=float, default=0.0, help='Whether to randomly zoom in for data augmentation')
parser.add_argument('--shear', type=float, default=0.0, help='Whether to randomly shear in for data augmentation')
parser.add_argument('--model', type=str, default="MobileNetV2", help='Your pre-trained classification model of choice')
args = parser.parse_args()


# Global settings
BATCH_SIZE = args.batch_size
WIDTH = args.resize_width
HEIGHT = args.resize_height
ALPHA = args.alpha

FC_LAYERS = [1024]
TRAIN_DIR = args.dataset + "/train/"
VAL_DIR = args.dataset + "/val/"
TEST_DIR= args.dataset + "/test/"
preprocessing_function = None
base_model = None

file_name_class_list=['none','blade_root','blade','blade_tip']


# Prepare the model
if args.model == "VGG16":
    from keras.applications.vgg16 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "VGG19":
    from keras.applications.vgg19 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "ResNet50":
    from keras.applications.resnet50 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "InceptionV3":
    from keras.applications.inception_v3 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "Xception":
    from keras.applications.xception import preprocess_input
    preprocessing_function = preprocess_input
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "InceptionResNetV2":
    from keras.applications.inceptionresnetv2 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "MobileNet":
    from keras.applications.mobilenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet121":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet169":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet201":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "NASNetLarge":
    from keras.applications.nasnet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = NASNetLarge(weights='imagenet', include_top=True, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "NASNetMobile":
    from keras.applications.nasnet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "MobileNetV2":
    from keras.applications.mobilenetv2 import preprocess_input
    preprocessing_function = preprocess_input
    from SqueezeNet import SqueezeNet
    base_model = MobileNetV2(weights='imagenet',alpha=ALPHA,include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "SqueezeNet":
    from keras.applications.nasnet import preprocess_input
    preprocessing_function = preprocess_input
    from SqueezeNet import SqueezeNet 
    base_model = SqueezeNet(weights=None, include_top=True, input_shape=(HEIGHT, WIDTH, 3))

else:
    ValueError("The model you requested is not supported in Keras")

    
if args.mode == "train":
    print("\n***** Begin training *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Resize Height -->", args.resize_height)
    print("Resize Width -->", args.resize_width)
    print("Num Epochs -->", args.num_epochs)
    print("Batch Size -->", args.batch_size)

    print("Data Augmentation:")
    print("\tVertical Flip -->", args.v_flip)
    print("\tHorizontal Flip -->", args.h_flip)
    print("\tRotation -->", args.rotation)
    print("\tZooming -->", args.zoom)
    print("\tShear -->", args.shear)
    print("")

    # Create directories if needed
    if not os.path.isdir("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.isdir("log"):
        os.makedirs("log")
    # Prepare data generators
    train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocessing_function,
      rotation_range=args.rotation,
      shear_range=args.shear,
      zoom_range=args.zoom,
      horizontal_flip=args.h_flip,
      vertical_flip=args.v_flip
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)

    validation_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)


    # Save the list of classes for prediction mode later
    class_list = utils.get_subfolders(TRAIN_DIR)
    utils.save_class_list(class_list, model_name=args.model, dataset_name=args.dataset)
    finetune_model = utils.build_finetune_model(base_model, dropout=args.dropout, fc_layers=FC_LAYERS, num_classes=len(class_list))
    if args.continue_training:
        finetune_model.load_weights("./checkpoints/" + args.model + "_model_weights.h5")
    
    #set_trainable_layer
    # base_model.trainable = True
    # set_trainable = False
    # for layer in base_model.layers:
    #     #print(layer.name)
    #     if layer.name == 'block_16_project':
    #         set_trainable = True
    #     if set_trainable:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False 
    print(finetune_model.summary())
    adam = Adam(lr=0.00001)
    finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
    num_train_images = utils.get_num_files(TRAIN_DIR)
    num_val_images = utils.get_num_files(VAL_DIR)

    def lr_decay(epoch):
        if epoch%20 == 0 and epoch!=0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr/2)
            print("LR changed to {}".format(lr/2))
        return K.get_value(model.optimizer.lr)

    learning_rate_schedule = LearningRateScheduler(lr_decay)

    filepath="./checkpoints/" + args.model + "_model_weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
    tb=TensorBoard("./log", histogram_freq=0, batch_size=args.batch_size, write_graph=True,
                write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [checkpoint,tb]


    history = finetune_model.fit_generator(train_generator, epochs=args.num_epochs, workers=8, steps_per_epoch=num_train_images // BATCH_SIZE, 
        validation_data=validation_generator, validation_steps=num_val_images // BATCH_SIZE, class_weight='auto', shuffle=True, callbacks=callbacks_list)


    utils.plot_training(history)

elif args.mode == "predict":

    if args.image is None:
        ValueError("You must pass an image path when using prediction mode.")

    # Create directories if needed
    if not os.path.isdir("%s"%("Predictions")):
        os.makedirs("%s"%("Predictions"))

    class_list_file = "./checkpoints/" + args.model + "_" + args.dataset + "_class_list.txt"    

    class_list = utils.load_class_list(class_list_file)
        
    finetune_model = utils.build_finetune_model(base_model, dropout=args.dropout, fc_layers=FC_LAYERS, num_classes=len(class_list))
    finetune_model.load_weights("./checkpoints/" + args.model + "_model_weights.h5")
    
    dict_acc = {}
    for class_id in class_list :
        dict_acc[class_id[0]] = 0

    # Read in your image
    for name in glob.glob(TEST_DIR+'*/*'):
	true_class=int(name.split("/")[-1].split('.')[0][-1])
	true_name=str(name.split("/")[-1].split('.')[0])
        image = cv2.imread(name,-1)
        save_image = image
        image = np.float32(cv2.resize(image, (HEIGHT, WIDTH)))
        image = preprocessing_function(image.reshape(1, HEIGHT, WIDTH, 3))

        # Run the classifier and print results
        st = time.time()

        out = finetune_model.predict(image)

        confidence = out[0]
        class_prediction = list(out[0]).index(max(out[0]))
        class_name = class_list[class_prediction][0]

        run_time = time.time()-st

        print("Predicted class = ", class_name)
        print("Confidence = ", confidence)
        print("Run time = ", run_time)
        #cv2.imwrite("Predictions/" + class_name[0]+str(time.time()) + ".png", save_image)
        _ , d = os.path.split(os.path.dirname(name))
        if d == class_name:
            dict_acc[class_name] = dict_acc[class_name] +1
        else :
            #pass
            cv2.imwrite("Predictions/" +class_name+"_"+str(time.time())+'_'+file_name_class_list[true_class]+ ".jpg" , save_image)
    dict_num = {}
    for class_id in class_list :
        dict_num[class_id[0]] = 0

    for class_name in class_list :
        for _, dirs, x in os.walk(TEST_DIR + class_name[0]) :
            dict_num[class_name[0]] = len(x)

    for key in dict_num:
        acc = float(dict_acc[key])/dict_num[key]
        print(key + " acc = " + str(acc))
