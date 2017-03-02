import numpy as np
import tensorflow as tf

sess = tf.Session()
def get_distance_matrix (train_dataset,test_dataset):

  x_expanded = tf.expand_dims(train_dataset,1)
  y_expanded = tf.expand_dims(test_dataset,0)

  z = tf.squared_difference(x_expanded,y_expanded)

  dist_matrix = tf.reduce_sum(z,2)
  return dist_matrix

def hard_resp(pw_matrix, k):
    '''
    hard_resp
        Calculates the standard KNN responsibility vector
    '''
   #We need to index the closest values
    ref_matrix = tf.neg(tf.transpose(pw_matrix))
    values, indices = tf.nn.top_k(ref_matrix, k, sorted=False)
    #Generate the indices from top_k (adapted liberally from Stack Overflow)
    range_repeated = tf.tile(tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1), [1, k])
    # Time to update
    full_indices = tf.reshape(tf.concat(2, [tf.expand_dims(range_repeated, 2),\
                             tf.expand_dims(indices, 2)]), [-1, 2])
    update = tf.mul(tf.truediv(tf.constant(1.0,dtype=tf.float64), \
                    tf.cast(k,tf.float64)),tf.ones(tf.shape(values), dtype=tf.float64))
    #update, and desparsify
    return tf.sparse_to_dense(full_indices, tf.shape(ref_matrix), tf.reshape(update, [-1]), \
                              default_value=0., validate_indices=False)

d_matrix = get_distance_matrix([[1.0],[4.4],[9.0]],[[9.2],[1.1],[2.0]])
print(sess.run(d_matrix))
resp = hard_resp(d_matrix,1)
print(sess.run(resp))

sess.close()  

