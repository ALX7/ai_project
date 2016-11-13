from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

NUM_CLASSES = 6

IMAGE_SIZE = 144
IMAGE_PIXELS = 500

def _variable_summaries(var, name):
	with tf.variable_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.scalar_summary('mean/' + name, mean)

		# with tf.variable_scope('stddev'):
		# 	stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

		# tf.scalar_summary('stddev/' + name, mean)
		# tf.scalar_summary('max/' + name, tf.reduce_max(var))
		# tf.scalar_summary('min/' + name, tf.reduce_min(var))
		tf.histogram_summary(name, var)

def inference(images, hidden1_units, hidden2_units, hidden3_units, hidden4_units, keep_prob=0.5):

	with tf.variable_scope('hidden1'):
		weights = tf.get_variable('weights', shape=[IMAGE_PIXELS,hidden1_units],
			initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable('biases', shape=[hidden1_units],
			initializer=tf.constant_initializer(0.1))
		hidden1 = tf.nn.relu(tf.matmul(images,weights) + biases)
		_variable_summaries(hidden1, 'hidden1')

	with tf.variable_scope('dropout1'):
		hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

	with tf.variable_scope('hidden2'):
		weights = tf.get_variable('weights', shape=[hidden1_units,hidden2_units],
			initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable('biases', shape=[hidden2_units],
			initializer=tf.constant_initializer(0.1))
		hidden2 = tf.nn.relu(tf.matmul(hidden1_drop,weights) + biases)
		_variable_summaries(hidden2, 'hidden2')

	with tf.variable_scope('dropout2'):
		hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

	with tf.variable_scope('hidden3'):
		weights = tf.get_variable('weights', shape=[hidden2_units,hidden3_units],
			initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable('biases', shape=[hidden3_units],
			initializer=tf.constant_initializer(0.1))
		hidden3 = tf.nn.relu(tf.matmul(hidden2_drop,weights) + biases)
		_variable_summaries(hidden3, 'hidden3')

	with tf.variable_scope('dropout3'):
		hidden3_drop = tf.nn.dropout(hidden3, keep_prob)

	with tf.variable_scope('hidden4'):
		weights = tf.get_variable('weights', shape=[hidden3_units,hidden4_units],
			initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable('biases', shape=[hidden4_units],
			initializer=tf.constant_initializer(0.1))
		hidden4 = tf.nn.relu(tf.matmul(hidden2_drop,weights) + biases)
		_variable_summaries(hidden4, 'hidden4')

	with tf.variable_scope('dropout4'):
		hidden4_drop = tf.nn.dropout(hidden4, keep_prob)

	with tf.variable_scope('softmax'):
		weights = tf.get_variable('weights', shape=[hidden4_units,NUM_CLASSES],
			initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable('biases', shape=[NUM_CLASSES],
			initializer=tf.constant_initializer(0.1))
		logits = tf.matmul(hidden2_drop, weights) + biases
		_variable_summaries(logits, 'softmax')

	return logits

def loss(logits, labels):

	labels = tf.to_int64(labels)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits, labels, name='cross-entropy')
	loss = tf.reduce_mean(cross_entropy, name='cross-entropy-mean')
	_variable_summaries(loss, 'loss')

	return loss

def training(loss, learning_rate):

	optimizer = tf.train.AdamOptimizer(learning_rate)

	global_step = tf.Variable(0, name='global_step', trainable=False)

	train_op = optimizer.minimize(loss, global_step=global_step)

	return train_op

def evaluation(logits, labels):

	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.to_int64(labels), tf.argmax(logits, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.scalar_summary('accuracy', accuracy)

	return accuracy
