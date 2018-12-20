import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format


def convert_pb_to_pbtxt(filename):
    
    filename_pbtxt = filename + '.pbtxt'
    filename = filename + '.pb'

    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()

        graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')

        tf.train.write_graph(graph_def, './', filename_pbtxt, as_text=True)
    return


def convert_pbtxt_to_pb(filename):
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    Args:
      filename: The name of a file containing a GraphDef pbtxt (text-formatted
        `tf.GraphDef` protocol buffer data).
    """
    filename_pb = filename + '.pb'
    filename = filename + '.pbtxt'
    
    with tf.gfile.FastGFile(filename, 'r') as f:
        graph_def = tf.GraphDef()

        file_content = f.read()

        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, './', filename_pb, as_text=False)
    return
    
def convert_pbtxt_str_to_pb(sss,path,pb_name):
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    Args:
      filename: The name of a file containing a GraphDef pbtxt (text-formatted
        `tf.GraphDef` protocol buffer data).
    """
    #filename_pb = filename + '.pb'
    #filename = filename + '.pbtxt'
    
    #with tf.gfile.FastGFile(filename, 'r') as f:
    graph_def = tf.GraphDef()

    #file_content = f.read()

    # Merges the human-readable string in `file_content` into `graph_def`.
    text_format.Merge(sss, graph_def)
    tf.train.write_graph(graph_def, path, pb_name, as_text=False)
    return