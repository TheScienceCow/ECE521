import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#generate data
np.random.seed(521)
Data = np.linspace(1.0 , 10.0, num=100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)

#Randomize what is test, what is training etc.
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]


def get_distance_matrix (train_dataset,test_dataset):

  x_expanded = tf.expand_dims(train_dataset,1)
  y_expanded = tf.expand_dims(test_dataset,0)

  z = tf.squared_difference(x_expanded,y_expanded)

  dist_matrix = tf.reduce_sum(z,2)
  return dist_matrix

def hard_resp(pw_matrix, k):
    '''
    hard_resp
        Calculates the hard KNN responsibility vector
    '''
   #We need to index the closest values
    ref_matrix = tf.neg(tf.transpose(pw_matrix))
    values, indices = tf.nn.top_k(ref_matrix, k, sorted=False)
    #Generate the indices from top_k (adapted liberally from Stack Overflow)
    range_repeated = tf.tile(tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1), [1, k])
    # Tiime to update
    full_indices = tf.reshape(tf.concat(2, [tf.expand_dims(range_repeated, 2), tf.expand_dims(indices, 2)]), [-1, 2])
    update = tf.mul(tf.truediv(tf.constant(1.0,dtype=tf.float64), tf.cast(k,tf.float64)),tf.ones(tf.shape(values), dtype=tf.float64))
    return tf.sparse_to_dense(full_indices, tf.shape(ref_matrix), tf.reshape(update, [-1]), default_value=0., validate_indices=False)

def knn(data, target, input, k):
    '''
    knn
        runs the KNN for a set of input values
    '''

    resp = hard_resp(get_distance_matrix(data, input), k)
    return tf.matmul(resp,target)

X = np.linspace(0.0, 11.0, num=1000)[:,np.newaxis]

K_var = tf.placeholder("int32") 
X_var = tf.placeholder("float64")
data = tf.placeholder("float64")
target = tf.placeholder("float64")

#run the KNN
Y_out = knn(data, target, X_var, K_var)

#The actual code.
for K in [1,3,5,50]:
    print("Running KNN for K = {0}".format(K))
    with tf.Session() as sess:
        Y = sess.run(Y_out, feed_dict={X_var: X, K_var: K, data: trainData, target: trainTarget})
        ValidationGuess = sess.run(Y_out, feed_dict={K_var: K, X_var: validData, data: trainData, target: trainTarget})
        TestGuess = sess.run(Y_out, feed_dict={K_var: K, X_var: testData, data: trainData, target: trainTarget})

    #Do validation/testing
    validMSE = np.sum(np.power(ValidationGuess-validTarget,2))/len(validData)/2
    TestMSE = np.sum(np.power(TestGuess-testTarget,2))/len(testData)/2
    print ("The MSE from the test data was {0}, while the MSE from the validation data was {1}".format(TestMSE,validMSE))

    #Plot the KNN as a function of input from 0-11
    plt.figure()
    plt.plot(X,Y, 'g')
    plt.plot(trainData,trainTarget, 'bo')
    plt.title('k-NN regression on data1D, k = {0}'.format(K))
plt.show()