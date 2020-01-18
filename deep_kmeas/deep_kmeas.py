#coding=utf-8
import random, cv2, os, sys, shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import glob
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2

class image_clustering:
    def __init__(self, folder_path="data", n_clusters=3, max_examples=None, use_imagenets=False, use_pca=False):
        paths = os.listdir(folder_path)
        if max_examples == None:
            self.max_examples = len(paths)
        else:
            if max_examples > len(paths):
                self.max_examples = len(paths)
            else:
                self.max_examples = max_examples
        self.n_clusters = n_clusters
        self.folder_path = folder_path
        random.shuffle(paths)
        self.image_paths = paths[:self.max_examples]
        self.use_imagenets = use_imagenets
        self.use_pca = use_pca
        # shutil.rmtree("output")
        #os.makedirs("output")
        #for i in range(self.n_clusters):
            #os.makedirs("output\\cluster" + str(i))
        print(" Object of class image_clustering has been initialized.")

    def load_images(self):
        self.images = []
        self.image_paths = glob.glob("./data/*")
        print(self.image_paths)
        for image_path in self.image_paths:
            self.images.append(
                cv2.cvtColor(cv2.resize(cv2.imread(image_path), (224, 224)), cv2.COLOR_BGR2RGB))
        self.images = np.float32(self.images).reshape(len(self.images), -1)
        self.images /= 255
    def get_new_imagevectors(self):
        if self.use_imagenets == False:
            self.images_new = self.images
        else:
            if use_imagenets.lower() == "vgg16":
                model1 = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
            elif use_imagenets.lower() == "vgg19":
                model1 = VGG19(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
            elif use_imagenets.lower() == "resnet50":
                model1 =  ResNet50(include_top=False, weights="imagenet",
                                                              input_shape=(224, 224, 3))
            elif use_imagenets.lower() == "xception":
                model1 = Xception(include_top=False, weights='imagenet',
                                                              input_shape=(224, 224, 3))
            elif use_imagenets.lower() == "mobilenetv2":
                model1 = MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, include_top=False,
                                                                    weights='imagenet', pooling=None)
            else:
                print("use a model please")
                sys.exit()
            print(self.images.shape)
            self.images = self.images.reshape(self.images.shape[0],224,224,3)
            pred = model1.predict(self.images)
            images_temp = pred.reshape(self.images.shape[0], -1)
            if self.use_pca == False:
                self.images_new = images_temp
            else:
                model2 = PCA(n_components=None, random_state=728)
                model2.fit(images_temp)
                self.images_new = model2
    def clustering(self):
        model = KMeans(n_clusters=self.n_clusters, n_jobs=-1, random_state=728)
        model.fit(self.images_new)
        predictions = model.predict(self.images_new)
        print(predictions)
        for i in range(self.max_examples):
            shutil.copy2(self.image_paths[i], "output\cluster" + str(predictions[i]))
        print("Clustering complete!")


if __name__ == "__main__":

    print("START")
    number_of_clusters = 15
    data_path = "data"
    max_examples = None  # number of examples to use, if "None" all of the images will be taken into consideration for the clustering
    use_imagenets = "VGG19"# choose from: "Xception", "VGG16", "VGG19", "ResNet50", "InceptionV3", "InceptionResNetV2", "DenseNet", "MobileNetV2" and "False" -> Default is: False
    if use_imagenets:
        use_pca = False
    else:
        use_pca = Trueue  # Make it True if you want to use PCA for dimentionaity reduction -> Default is: False
    temp = image_clustering(data_path, number_of_clusters, max_examples, use_imagenets, use_pca)
    temp.load_images()
    temp.get_new_imagevectors()
    temp.clustering()
    print("END")
