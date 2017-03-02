import tensorflow as tf

def get_distance_matrix (train_dataset,test_dataset):

  sess = tf.Session()
  x = tf.placeholder(tf.int64,[None,None])
  y = tf.placeholder(tf.int64,[None,None])

  x_expanded = tf.expand_dims(x,1)
  y_expanded = tf.expand_dims(y,0)

  z = tf.squared_difference(x_expanded,y_expanded)

  dist_matrix = tf.reduce_sum(z,2)
  out = sess.run(dist_matrix,feed_dict={x:train_dataset, y:test_dataset})
  sess.close()
  
  return out

def main():
  print(get_distance_matrix([[1,2,3],[4,5,6],[7,8,9]],[[9,8,7],[6,5,4],[3,2,1]]))
  
if __name__ == "__main__":
    main()
