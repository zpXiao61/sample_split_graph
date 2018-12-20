import tensorflow as tf
from tensorflow.python.platform import gfile
import time

def run_tf_pb(PATH_TO_FROZEN_GRAPH,itensors_name,ivalues,otensors_name,warm_loop,loop):
    assert(warm_loop >= 0)
    assert(loop > 0)
    assert(len(itensors_name) == len(ivalues))
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        
        itensors = []
        for t in itensors_name:
            itensors.append(tf_graph.get_tensor_by_name(t))
            
        input_dict = dict(zip(itensors,ivalues))
        
        otensors = []
        
        for t in otensors_name:
            otensors.append(tf_graph.get_tensor_by_name(t))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(warm_loop):
                rets = sess.run(otensors, feed_dict=input_dict) 
            start = time.time()
            for i in range(loop):
                rets = sess.run(otensors, feed_dict=input_dict)    
            end = time.time()
            
    return rets,(end-start)/loop
        
