import tensorflow as tf
import numpy as np
import random
import os
import sys
import argparse
from tensorflow.python.platform import gfile
from PIL import Image
import imutils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def random_rotate(batch):
    angle = random.randint(-10,10)

    for i in range(np.shape(batch)[0]):
        batch[i] = imutils.rotate(batch[i], angle)
    return batch

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = random_rotate(batch)
    batch = _random_crop(batch, [160, 160], 4)
    return batch

class FeatureExtractor():
    def draw_graph_pb(self, model_path):
        with tf.Graph().as_default():
            model_exp = os.path.expanduser(model_path)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            # Get input and output tensors
            sess = tf.Session()
            input_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            input_op = [sess, input_placeholder, phase_train_placeholder]
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        return input_op, embeddings

    def run_graph(self, input_op, embeddings, npzfile):

        sess, input_placeholder, phase_train_placeholder = input_op

        feed_dict = {input_placeholder: npzfile, phase_train_placeholder: False}
        output_vector = sess.run(embeddings, feed_dict=feed_dict)

        return output_vector



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str,
        help='Path to the pretrained feature extractor.')
    parser.add_argument('--path_imgs', type=str,
        help='Path to the data directory containing face images.')
    parser.add_argument('--res_name', type=str,
        help='Path to the result npz.')
    return parser.parse_args(argv)


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])

    img_all = []

    for root, subdirs, files in os.walk(args.path_imgs):
        for file in files:
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png'):
                print(os.path.join(root, file))
                img = np.array(Image.open(os.path.join(root, file)))
                img_all.append(img)

    for i in range(10):
        if i == 0:
            img_aug = data_augmentation(img_all)
        else:
            img_aug = np.concatenate((img_aug, data_augmentation(img_all)), axis=0)

    print(np.shape(img_aug))


    FE = FeatureExtractor()

    input_op, embeddings = FE.draw_graph_pb(args.pretrained_model)
    output_vector = FE.run_graph(input_op, embeddings, img_aug)
    print(np.shape(output_vector))

    np.savez_compressed(args.res_name, output_vector)
