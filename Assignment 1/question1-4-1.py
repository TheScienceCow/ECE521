import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#parameters
Lambda=10
GaussianSigma2e = 0.25 #To make sure the solution doesn't diverge, we include the possible error on each point

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

def soft_resp_knn(pw_matrix, l):
    '''
    soft_resp
        Calculates the soft KNN responsibility vector
    '''
    #We need to index the closest values
    ref_matrix = tf.exp(l*tf.neg(tf.transpose(pw_matrix)))
    ref_sums = tf.reduce_sum(ref_matrix,axis=1)
    ref_sums_exp = tf.expand_dims(ref_sums,1)
    ref_sums = tf.tile( ref_sums_exp, (1,tf.shape(ref_matrix)[1]))
    #Generate the indices    
    return tf.truediv(ref_matrix,ref_sums)   

def sqk_gpr(input1, input2, l):
    '''
    sqk_gpr
        Calculates the squared exponential kernel
    '''
    #We need to index the closest values
    return tf.exp(l*tf.neg(get_distance_matrix(input1,input2)))

def knn(train, target, input, l):
    '''
    knn
        runs the KNN for a set of input values
    '''
    resp = soft_resp_knn(get_distance_matrix(train, input), l)
    return tf.matmul(resp,target)

def gpr(train, target, input, l, sigma2e):
    '''
    gpr
        runs Gaussian process regression using a set of input values
    '''
    sigma2e.set_shape([])
    left_m= tf.matrix_inverse(tf.add(sqk_gpr(train, train, l),tf.scalar_mul(sigma2e,tf.eye(tf.shape(train)[0], dtype=tf.float64))))
    resp = tf.matmul(left_m, sqk_gpr(train, input, l))
    return tf.matmul(target, resp,True, False)    


L_var = tf.placeholder("float64") 
X_var = tf.placeholder("float64")
data = tf.placeholder("float64")
target = tf.placeholder("float64")
sigma2e = tf.placeholder("float64")

#build the KNN
Y_out_k = knn(data, target, X_var, L_var)
#build the GPR
Y_out_g = gpr(data, target, X_var, L_var, sigma2e)


#Populate and run the three graphs
X = np.linspace(0, 11.0, num=1000)[:,np.newaxis]

with tf.Session() as sess:
    KNN_Y = sess.run(Y_out_k, feed_dict={X_var: X, L_var: Lambda, data: trainData, target: trainTarget, sigma2e: GaussianSigma2e})
    KNN_ValidationGuess = sess.run(Y_out_k, feed_dict={X_var: validData, L_var: Lambda, data: trainData, target: trainTarget, sigma2e: GaussianSigma2e})
    KNN_TestGuess = sess.run(Y_out_k, feed_dict={X_var: testData, L_var: Lambda, data: trainData, target: trainTarget, sigma2e: GaussianSigma2e})
    GPR_Y = sess.run(Y_out_g, feed_dict={X_var: X, L_var: Lambda, data: trainData, target: trainTarget, sigma2e: GaussianSigma2e})
    GPR_ValidationGuess = sess.run(Y_out_g, feed_dict={L_var: Lambda, X_var: validData, data: trainData, target: trainTarget, sigma2e: GaussianSigma2e})
    GPR_TestGuess = sess.run(Y_out_g, feed_dict={L_var: Lambda, X_var: testData, data: trainData, target: trainTarget, sigma2e: GaussianSigma2e})

#Do validation/testing
validMSE = np.sum(np.power(KNN_ValidationGuess - validTarget,2))/len(validData)/2
TestMSE = np.sum(np.power(KNN_TestGuess - testTarget,2))/len(testData)/2
print ("The KNN MSE from the test data was {0}, while the MSE from the validation data was {1}".format(TestMSE,validMSE))

validMSE = np.sum(np.power(GPR_ValidationGuess - validTarget,2))/len(validData)/2
TestMSE = np.sum(np.power(GPR_TestGuess - testTarget,2))/len(testData)/2
print ("The GPR MSE from the test data was {0}, while the MSE from the validation data was {1}".format(TestMSE,validMSE))

#Plot the regressions as a function of input from 0-11
plt.figure()
plt.plot(trainData, trainTarget, 'bo')
plt.plot(X,KNN_Y, 'g')
plt.title('Soft k-NN on data1D, lambda = {0}'.format(Lambda))
plt.figure()
plt.plot(trainData, trainTarget, 'bo')
plt.plot(X,np.transpose(GPR_Y), 'g')
plt.title('Gaussian process regression on data1D, lambda = {0}'.format(Lambda))
plt.show()

sess.close()  