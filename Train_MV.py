import tensorflow as tf
import numpy as np
import argparse
import os
import sys

from tensorflow.contrib.layers import xavier_initializer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

initializer = xavier_initializer()
threshold = 0.5
NB_CLASSES = 2

with tf.variable_scope("placeholder"):
    x = tf.placeholder(tf.float32, shape=[None, 512], name='x')
    label = tf.placeholder(tf.float32, shape=[None, 2], name='label')
    training_flag = tf.placeholder(tf.bool, name='phase_train')
    global_step = tf.placeholder(tf.float32, name='global_step')


class MV():
    def __init__(self, training):
        self.training = training

    def Model(self, input_x):
        x = Linear(input_x, 256, layer_name='fc0')
        x = Relu(x)

        x = Linear(x, 128, layer_name='fc1')
        x = Relu(x)

        logits = Linear(x, 2, layer_name='fc2')
        confidence = tf.nn.softmax(logits, name='confidence')
        is_True = tf.greater(confidence, threshold)
        is_True = tf.split(is_True, 2, axis=1)
        result = tf.identity(is_True[0], name='result')

        return confidence

def main(args):
    embeddings_M = np.load(args.members)['arr_0']
    embeddings_N = np.load(args.other_members)['arr_0']
    embeddings_O = np.load(args.non_members)['arr_0']
    embeddings_N = np.concatenate((embeddings_N, embeddings_O), axis=0)

    num_M = np.shape(embeddings_M)[0]
    num_N = np.shape(embeddings_N)[0]

    embeddings = np.concatenate((embeddings_M, embeddings_N), axis=0)
    labels = np.concatenate((np.zeros(num_M, dtype=int),np.ones(num_N, dtype=int)), axis=0)
    labels = np.eye(NB_CLASSES)[labels]

    Net = MV(training=training_flag)

    logits = Net.Model(x)

    cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    cost = cls_loss

    train_op = tf.train.AdamOptimizer().minimize(loss=cost)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_node_names = ['confidence', 'result']

    total_epochs = 500
    batch_size = 200
    num_img = num_M + num_N
    iteration = int(num_img / batch_size)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, total_epochs+1):

            for step in range(iteration):

                rnd_id_A = np.random.choice(num_M, size=int(batch_size/2), replace=False)
                rnd_id_B = np.random.choice(num_N, size=int(batch_size/2), replace=False)

                rnd_id_B += num_M
                rnd_id = np.concatenate((rnd_id_A, rnd_id_B), axis=0)
                batch_x = embeddings[rnd_id]
                batch_y = labels[rnd_id]

                _, batch_logits, batch_loss = sess.run([train_op, logits, cost], feed_dict={x: batch_x, label: batch_y, training_flag: True})
                batch_train_acc = accuracy.eval(feed_dict={x: batch_x, label: batch_y, training_flag: False})
                print("epoch: %d, iter: %d - loss: %.4f, train_acc: %.4f" % (epoch ,step, batch_loss, batch_train_acc))

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)

        with tf.gfile.GFile(args.output_model, 'wb') as f:
            f.write(output_graph_def.SerializeToString())


def Linear(x, out_length, layer_name) :
    with tf.name_scope(layer_name):
        linear = tf.layers.dense(inputs=x, units=out_length, kernel_initializer=initializer)
        return linear

def Relu(x):
    return tf.nn.relu(x)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', type=str,
        help='Path to the data directory containing member npz.')
    parser.add_argument('--other_members', type=str,
        help='Path to the data directory containing other_member npz.')
    parser.add_argument('--non_members', type=str,
        help='Path to the data directory containing non_member npz.')
    parser.add_argument('--output_model', type=str,
        help='Path to the result verifier model.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
