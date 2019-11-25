#coding=utf-8
#transfet keras model .h5 to tensorflow model .pb
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from keras import backend as K
import os
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Activation, Flatten, Dropout,GlobalAveragePooling2D
 
 
base_model = MobileNetV2((None, None, 3), alpha=0.35, include_top=False,  weights=None)
#x = Dropout(0.75)(base_model.output)
gap_output=GlobalAveragePooling2D()(base_model.output)
predictions = Dense(4, activation='softmax')(gap_output)
model = Model(base_model.input, predictions)
model.load_weights('MobileNetV2_model_weights.h5')
 
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                    output_names, freeze_var_names)
        return frozen_graph
 
output_graph_name = 'new.pb'
output_fld = ''
#K.set_learning_phase(0)
 
print('input is :', model.input.name)
print ('output is:', model.output.name)
 
sess = K.get_session()
frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])
 
from tensorflow.python.framework import graph_io
graph_io.write_graph(frozen_graph, output_fld, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', os.path.join(output_fld, output_graph_name))

