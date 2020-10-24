# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 14:08:08 2020

@author: Administrator
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

X_TRUE = np.linspace(0, 100, 100)[:, np.newaxis].astype('float32')
Y_TRUE = np.subtract(np.cos(X_TRUE), np.sin(X_TRUE)).astype('float32')

def RBF(X):
    return tf.exp(tf.multiply(-0.5, tf.square(X)))

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

L1 = add_layer(X_TRUE, 1, 10)
Y_PRED = add_layer(L1, 10, 1, RBF)

LOSS = tf.reduce_mean( tf.reduce_sum(
    tf.square(Y_TRUE - Y_PRED), axis=[1] ) )

TRAIN_STEP = tf.train.GradientDescentOptimizer(0.1).minimize(LOSS)

INIT = tf.initialize_all_variables()
with tf.Session() as SS:
    SS.run(INIT)
    for i in range(1000):
        SS.run(TRAIN_STEP)
        if i % 50 == 0:
            print(SS.run(LOSS))