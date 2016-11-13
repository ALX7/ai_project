from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import os
from sklearn.decomposition import IncrementalPCA

import model as model

BATCH_SIZE = 128

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 4001, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 1000, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 1000, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('summaries_dir', 'summary', 'Directory to put the training data.')
flags.DEFINE_string('suffix', 'normal', 'Whether to use the augmented data or not')
flags.DEFINE_string('result_file', 'results.txt', 'Name of the file which records the results')

def run_training():

	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable=False)

		images = np.load('X_%s.npy' % FLAGS.suffix)
		labels = np.load('y_%s.npy' % FLAGS.suffix)

		print("Dimensions of input:")
		print(images.shape, labels.shape)

		ratio = 0.7
		X_train = images[:int(images.shape[0]*ratio)]
		y_train = labels[:int(labels.shape[0]*ratio)]

		X_test = images[int(images.shape[0]*ratio):]
		y_test = labels[int(labels.shape[0]*ratio):]

		X_mean = np.mean(X_train, axis=0)

		X_train -= X_mean
		X_test -= X_mean

		ipca = IncrementalPCA(n_components=500, whiten=True)
		ipca.fit(X_train)
		X_train = ipca.transform(X_train)
		X_test = ipca.transform(X_test)

		X = tf.placeholder(tf.float32, shape=(None, model.IMAGE_PIXELS))
		y = tf.placeholder(tf.int32, shape=(None))
		keep_prob = tf.placeholder(tf.float32)

		logits = model.inference(X, FLAGS.hidden1, FLAGS.hidden2, 1000, 1000,
			keep_prob=keep_prob)

		loss = model.loss(logits, y)

		train_op = model.training(loss, FLAGS.learning_rate)

		accuracy = model.evaluation(logits, y)

		summary = tf.merge_all_summaries()

		init = tf.initialize_all_variables()

		saver = tf.train.Saver()

		sess = tf.Session()

		summary_writer = tf.train.SummaryWriter(FLAGS.summaries_dir, sess.graph)

		sess.run(init)

		for step in xrange(FLAGS.max_steps):
			start_time = time.time()

			indices = np.random.choice(X_train.shape[0], FLAGS.batch_size)
			X_batch = X_train[indices]
			y_batch = y_train[indices]

			_, loss_value = sess.run([train_op, loss],
				feed_dict={X:X_batch, y:y_batch, keep_prob:0.5})

			duration = time.time() - start_time

			f = open("results/%s.txt" % FLAGS.result_file,"a")

			if step % 100 == 0:
				train_accuracy = sess.run(accuracy, feed_dict={X:X_batch, y:y_batch, keep_prob:0.5})
				results_str = 'Step %d: loss = %.7f accuracy: %.3f (%.3f sec)' % (step, loss_value,train_accuracy,duration)
				f.write(results_str + '\n')
				print(results_str)

				summary_str = sess.run(summary,
					feed_dict={X:X_batch, y:y_batch, keep_prob:0.5})
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

				if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
					checkpoint_file = os.path.join(FLAGS.summaries_dir, 'checkpoint_%s' % FLAGS.suffix)
					saver.save(sess, checkpoint_file, global_step=step)

		test_accuracy = sess.run(accuracy, feed_dict={X:X_test, y:y_test, keep_prob:1.0})
		test_acc_str = 'Testing accuracy: %.3f' % test_accuracy
		f.write(test_acc_str + '\n')
		print(test_acc_str)

def main(argv=None):
	if tf.gfile.Exists(FLAGS.summaries_dir):
		tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
	tf.gfile.MakeDirs(FLAGS.summaries_dir)
	run_training()

if __name__ == '__main__':
	tf.app.run()


