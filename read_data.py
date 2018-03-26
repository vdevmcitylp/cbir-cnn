import numpy as np

from sklearn import preprocessing

import itertools

def c10():

	num_class = 100
	num_train_per_class = 80
	num_test_per_class = 20

	train_file = "/home/rohit/pro/Features/C10K/80-20/c10_cnn_train.npy"
	test_file = "/home/rohit/pro/Features/C10K/80-20/c10_cnn_test.npy"

	x_train = np.load(open(train_file))
	x_test = np.load(open(test_file))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_train_per_class)
	y_train = np.array(list(itertools.chain.from_iterable(labels)))
	y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(num_class)))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_test_per_class)
	y_test = np.array(list(itertools.chain.from_iterable(labels)))
	y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(num_class)))

	return x_train, y_train, x_test, y_test
	# return y_train, y_test

def c1():

	num_class = 10
	num_train_per_class = 60
	num_test_per_class = 40

	train_file = "/home/rohit/pro/Features/C1K/90-10/c1_cnn_train.npy"
	test_file = "/home/rohit/pro/Features/C1K/90-10/c1_cnn_test.npy"

	print "OK"

	x_train = np.load(open(train_file))
	x_test = np.load(open(test_file))

	# labels = []
	# for i in xrange(num_class):
	# 	labels.append([i] * num_train_per_class)
	# y_train = np.array(list(itertools.chain.from_iterable(labels)))
	# y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(num_class)))

	# labels = []
	# for i in xrange(num_class):
	# 	labels.append([i] * num_test_per_class)
	# y_test = np.array(list(itertools.chain.from_iterable(labels)))
	# y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(num_class)))

	# return x_train, y_train, x_test, y_test
	# return y_train, y_test
	print x_train.shape
	return x_train, x_test

def pascal():

	num_class = 20
	num_train_per_class = 350
	num_test_per_class = 150

	train_file = "/home/rohit/pro/Features/Pascal/70-30/pascal_cnn_train.npy"
	test_file = "/home/rohit/pro/Features/Pascal/70-30/pascal_cnn_test.npy"

	# x_train = np.load(open(train_file))
	# x_test = np.load(open(test_file))

	y_train = np.load(open('/home/rohit/pro/Features/Pascal/70-30/pascal_labels_train.npy'))
	y_test = np.load(open('/home/rohit/pro/Features/Pascal/70-30/pascal_labels_test.npy'))

	return y_train, y_test


def ghim():

	num_class = 20
	num_train_per_class = 400
	num_test_per_class = 100

	train_file = "/home/rohit/pro/Features/GHIM-10K/80-20/ghim_cnn_train.npy"
	test_file = "/home/rohit/pro/Features/GHIM-10K/80-20/ghim_cnn_test.npy"

	x_train = np.load(open(train_file))
	x_test = np.load(open(test_file))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_train_per_class)
	y_train = np.array(list(itertools.chain.from_iterable(labels)))
	y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(num_class)))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_test_per_class)
	y_test = np.array(list(itertools.chain.from_iterable(labels)))
	y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(num_class)))

	return x_train, y_train, x_test, y_test
	# return y_train, y_test	


def preprocess(x_train, x_test):

	x_train = preprocessing.scale(x_train)
	x_test = preprocessing.scale(x_test)

	return x_train, x_test