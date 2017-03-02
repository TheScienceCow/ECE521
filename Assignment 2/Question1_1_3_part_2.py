import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

y_pred = tf.placeholder(tf.float32, [None,1])
y_target = tf.placeholder(tf.float32, [None, 1])

squared_error = tf.pow(tf.subtract(y_pred, y_target), 2)

cross_entropy = -tf.log( 1 - y_pred )

sess = tf.Session()
sess.run(tf.global_variables_initializer())

y = np.linspace(0.0, 1.0, num=20)[:, np.newaxis]
y_tar = np.zeros(20)[:, np.newaxis]

cross_entropy_loss = sess.run(cross_entropy, feed_dict={y_target: y_tar, y_pred: y})
squared_error_loss = sess.run(squared_error, feed_dict={y_target: y_tar, y_pred: y})

x = list(range(1, len(cross_entropy_loss)+1))
plt.plot(y, cross_entropy_loss, label = 'Cross Entropy' )
plt.plot(y, squared_error_loss, label='Squared Error')
plt.ylabel('Loss')
plt.xlabel('Prediction')
plt.title('Cross-entropy vs Squared-error')
plt.legend()
plt.show()
