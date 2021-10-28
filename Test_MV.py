import tensorflow as tf
import numpy as np
import os
import argparse
import sys
import json
from tensorflow.python.platform import gfile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_imgs', type=str,
        help='Path to the data directory containing test npz.')
    parser.add_argument('--MV_A', type=str,
        help='Path to the model for A members.')
    parser.add_argument('--MV_B', type=str,
        help='Path to the model for B members.')
    return parser.parse_args(argv)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class MemberVerifier():
    def draw_graph_pb(self, model_path):
        with tf.Graph().as_default():
            model_exp = os.path.expanduser(model_path)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            # Get input and output tensors
            sess = tf.Session()
            input_placeholder = tf.get_default_graph().get_tensor_by_name('placeholder/x:0')
            #phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('placeholder/phase_train:0')
            input_op = [sess, input_placeholder]
            confidence = tf.get_default_graph().get_tensor_by_name('confidence:0')
            result = tf.get_default_graph().get_tensor_by_name('result:0')

        return input_op, confidence, result

    def run_graph(self, input_op, confidence, result, npzfile, name):
        sess, input_placeholder = input_op
        feed_dict = {input_placeholder: npzfile}
        out_confidence, out_result = sess.run([confidence, result], feed_dict=feed_dict)
        json_out ={'groupName': name, 'confidence': out_confidence, "result": out_result}

        return json.dumps(json_out, cls=NumpyEncoder)

    def evaluate(self, A_result, B_result, threshold=0.9):

        A_result = json.loads(A_result)
        B_result = json.loads(B_result)

        A_con = A_result['confidence']
        B_con = B_result['confidence']
        print(A_con[3], B_con[3])

        for i in range(np.shape(A_con)[0]):
            if A_con[i][0] < threshold and B_con[i][0] <threshold:
                print("Stranger")
            else:
                if A_con[i][0] > B_con[i][0]:
                    print("Member A")
                else:
                    print("Member B")

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    #Load example npz
    vec = np.load(args.test_imgs)['arr_0']

    MV = MemberVerifier()

    input_op, confidence, result = MV.draw_graph_pb(args.MV_A)
    A_result = MV.run_graph(input_op, confidence, result, vec, 'A')

    input_op, confidence, result = MV.draw_graph_pb(args.MV_B)
    B_result = MV.run_graph(input_op, confidence, result, vec, 'B')

    MV.evaluate(A_result, B_result)


