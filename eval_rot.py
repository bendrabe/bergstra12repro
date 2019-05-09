import numpy as np
import os
import random
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import json

winit_dict = {
	"uniform_fanin": tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True),
	"uniform_fanavg": tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True),
	"norm_fanin": tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
	"norm_fanavg": tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
}
act_dict = {"sigmoid": tf.nn.sigmoid, "tanh": tf.nn.tanh}

out_dir = "summary/rand_rot/"
tmp_dir = "summary/tmp_rot/"

def nn_model_fn(features, labels, mode, params):
	input_layer = tf.reshape( features["x"], [-1, 28*28] )
	dense = tf.layers.dense(inputs=input_layer, units=params['hu'], activation=params['act'], kernel_initializer=params['winit'], kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2']))
	logits = tf.layers.dense(inputs=dense, units=10, kernel_initializer=tf.zeros_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2']))
	l2_loss = tf.losses.get_regularization_loss()

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) + l2_loss

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['lr_tensor'])
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])
	}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

tf.logging.set_verbosity(tf.logging.ERROR)

data_dir="/home/brabe2/ece543/data/rot/"

train_data = np.load(data_dir+"td_np.npy")
train_labels = np.load(data_dir+"tl_np.npy")

valid_data = np.load(data_dir+"vd_np.npy")
valid_labels = np.load(data_dir+"vl_np.npy")

test_data = np.load(data_dir+"Td_np.npy")
test_labels = np.load(data_dir+"Tl_np.npy")

for s in range(64):
	ea = event_accumulator.EventAccumulator(out_dir+str(s)+"/eval/")
	ea.Reload()
	s_acc = pd.DataFrame(ea.Scalars('accuracy'))['value'].max()
	diff_l = []
	hyper_l = []
	with open(out_dir+"results.txt", "r") as f:
		for line in f:
			i_hyper, i_acc = line.split('},')
			i_acc = float(i_acc)
			i_hyper += '}'
			diff_l.append( abs(s_acc - i_acc) )
			hyper_l.append(i_hyper)

	diff_p = np.array( diff_l )
	hyper_p = np.array( hyper_l )
	p = diff_p.argsort()
	hyper_sort = hyper_p[p]
	done = False
	i = 0
	while i < len(hyper_l) and not done:
		hyper = hyper_sort[i]
		hyper = hyper.replace('\'', '"')
		hyper = json.loads(hyper)

		print(hyper)

		hyper['winit'] = winit_dict[hyper['winit']]
		hyper['act'] = act_dict[hyper['act']]

		DO_NOT_USE_classifier = tf.estimator.Estimator(model_fn=nn_model_fn, model_dir=out_dir+str(s), params=hyper)
		latest_checkpoint = DO_NOT_USE_classifier.latest_checkpoint()
		mnist_classifier = tf.estimator.Estimator(model_fn=nn_model_fn, model_dir=tmp_dir+str(s), params=hyper)

		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": valid_data},
			y=valid_labels,
			num_epochs=1,
			shuffle=False)

		test_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": test_data},
			y=test_labels,
			num_epochs=1,
			shuffle=False)

		try:
			eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn, checkpoint_path=latest_checkpoint)
			val = eval_results['accuracy']

			test_results = mnist_classifier.evaluate(input_fn=test_input_fn, checkpoint_path=latest_checkpoint)
			test = test_results['accuracy']

			with open(out_dir+"eval_exp.txt", "a") as f:
				f.write(str(s) + "," + str(val) + "," + str(test) + "\n")
			done = True
		except:
			i += 1
			print("missed one")
