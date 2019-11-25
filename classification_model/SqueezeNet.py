
import keras.backend as K

from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/wohlert/keras-squeezenet/releases/download/v0.1/squeezenet_weights.h5'

# define some auxiliary variables and the fire module
sq1x1  = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu   = "relu_"


def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x     = Activation('relu', name=s_id + relu + sq1x1)(x)

    left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left  = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=3, name=s_id + 'concat')

    return x

def SqueezeNet(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling="avg", classes=1000,**kwargs):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name='conv1')(img_input)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1', padding="valid")(x)

    # x = _fire(x, (16, 64, 64), name="fire2")
    # x = _fire(x, (16, 64, 64), name="fire3")

    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3', padding="valid")(x)

    # x = _fire(x, (32, 128, 128), name="fire4")
    # x = _fire(x, (32, 128, 128), name="fire5")

    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool5', padding="valid")(x)

    # x = _fire(x, (48, 192, 192), name="fire6")
    # x = _fire(x, (48, 192, 192), name="fire7")

    # x = _fire(x, (64, 256, 256), name="fire8")
    # x = _fire(x, (64, 256, 256), name="fire9")
 


    # define the model of SqueezeNet

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=32)
    x = fire_module(x, fire_id=3, squeeze=16, expand=32)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=64)
    x = fire_module(x, fire_id=5, squeeze=32, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=128)
    x = fire_module(x, fire_id=7, squeeze=48, expand=128)
    x = fire_module(x, fire_id=8, squeeze=64, expand=128)
    x = fire_module(x, fire_id=9, squeeze=64, expand=128)

    if include_top:
        x = Dropout(0.5, name='dropout9')(x)
        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    #     x = AveragePooling2D(pool_size=(13, 13), name='avgpool10')(x)
    #     x = Flatten(name='flatten10')(x)
    #    # x = Activation("softmax", name='softmax')(x)
    # else:
    #     if pooling == "avg":
    #         x = GlobalAveragePooling2D(name="avgpool10")(x)
    #     else:
    #         x = GlobalMaxPooling2D(name="maxpool10")(x)

    model = Model(img_input, x, name="squeezenet")

    if weights == 'imagenet':
        # weights_path = get_file('squeezenet_weights.h5',
        #                         WEIGHTS_PATH,
        #                         cache_subdir='models')

        #model.load_weights('./squeezenet_weights.h5')
        print(model.summary())

    return model


