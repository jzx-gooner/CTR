#coding=utf-8
import tensorflow as tf

sess=tf.Session 
with tf.Graph().as_default(): 
	with tf.gfile.FastGFile("new.pb","rb") as modelfile: 
		graph_def=tf.GraphDef() 
		graph_def.ParseFromString(modelfile.read()) 
		tf.import_graph_def(graph_def) 
		for n in tf.get_default_graph().as_graph_def().node:
			print(n.name) 

