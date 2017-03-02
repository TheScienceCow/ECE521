import tensorflow as tf

def get_distance_matrix (train_dataset,test_dataset):

  x_expanded = tf.expand_dims(train_dataset,1)
  y_expanded = tf.expand_dims(test_dataset,0)

  z = tf.squared_difference(x_expanded,y_expanded)
  return tf.reduce_sum(z,2)

def main():
  
  train = tf.placeholder(tf.int64,[None,None])
  test = tf.placeholder(tf.int64,[None,None])

  dmatrix = get_distance_matrix(train, test)
  train_in =  [[9,1],[1,2],[1,3]]
  test_in = [[1,2],[4,12]]

  sess = tf.Session()
  print(sess.run(dmatrix,feed_dict={train:train_in, test:test_in}))

  sess.close()
  


if __name__ == "__main__":
    main()
