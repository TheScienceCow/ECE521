import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
rng = np.random
np.random.seed(0)

def trainSGD(miniBatchSize,learningRate,weightDecay,trainingEpochs):

  with np.load ("tinymnist.npz") as data :
    trainData, trainTarget = data ["x"], data["y"]
    validData, validTarget = data ["x_valid"], data ["y_valid"]
    testData, testTarget = data ["x_test"], data ["y_test"]
    
    numTrainingSet = len(trainTarget)
    
    X = tf.placeholder(tf.float64,[None,64])
    Y = tf.placeholder(tf.float64,[None,1])
    
    W = tf.Variable(rng.randn(64), name="weight")
    b = tf.Variable(rng.randn(1), name="bias")

    pred = tf.reshape((tf.add(tf.reduce_sum(tf.mul(X, W),1), b)),[-1,1])
    class_pref = tf.round(tf.div(tf.add(tf.sign(tf.subtract(pred,0.5)),1),2))
    accuracy = 1 - tf.reduce_mean(tf.abs(tf.subtract(class_pref,Y)))
    
    cost = tf.reduce_sum(tf.pow(tf.subtract(pred,Y), 2))/(2*numTrainingSet) + (weightDecay/2)* tf.reduce_sum(tf.pow(W,2))
    
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    updates = []
    validation_loss_list = []
    
    with tf.Session() as sess:
      sess.run(init)
      
      # Fit all training data
      for epoch in range(trainingEpochs):
        randIndex = np.arange(numTrainingSet)
        rng.shuffle(randIndex)
        shuffledData = trainData[randIndex]
        shuffledTarget = trainTarget[randIndex]
        start = 0
        for i in range(numTrainingSet//miniBatchSize):
          sess.run(optimizer, feed_dict={X: shuffledData[start:start+miniBatchSize], Y: shuffledTarget[start:start+miniBatchSize]})
          start = start + miniBatchSize
          
        if epoch % 100 == 0:
          print(epoch//100)
          print("Cost: ", sess.run(cost, feed_dict={X: trainData, Y: trainTarget}))
          print("Training Accuracy: ", sess.run(accuracy, feed_dict={X: trainData, Y: trainTarget}))
        updates.append(epoch)
        validation_loss_list.append(sess.run(cost, feed_dict={X: validData, Y: validTarget}))
            
      print("Finished Optimization")    
      print("Final test accuracy", sess.run(accuracy,feed_dict={X: testData, Y: testTarget}))
      
      return updates, validation_loss_list, sess.run(accuracy,feed_dict={X: testData, Y: testTarget})

def main():
    num_epoch = 200

    plt.figure(1)
    for learning_rate in [0.00001,0.0001,0.001,0.01,0.1,1]:
      updates,validation_loss_list,test_accuracy = trainSGD(50,learning_rate,1,num_epoch)
      plt.plot(updates,validation_loss_list,label = "Learning Rate = {0}".format(learning_rate))
    plt.legend(loc = "upper right")
    plt.xlabel("Iterations")
    plt.ylabel("Validation Loss")
    plt.title("Tuning the Learning Rate")
    '''
    plt.figure(2)
    for batch_size in [10, 50, 100, 700]:
      updates,validation_loss_list,test_accuracy = trainSGD(batch_size,0.01,1,num_epoch)
      plt.plot(updates,validation_loss_list,label = "Batch Size = {0}".format(batch_size))
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Validation Loss")
    plt.title("Tuning the Batch Size")

    fig3 = plt.figure(3)
    weight_decay_list = [0., 0.0001, 0.001, 0.01, 0.1, 1.]
    test_accuracy_list = []
    for weight_decay in weight_decay_list:
      updates,validation_loss_list,test_accuracy = trainSGD(50,0.01,weight_decay,1000)
      test_accuracy_list.append(test_accuracy)
    plt.plot(weight_decay_list,test_accuracy_list,"ro")
    plt.xlabel("Weight Decay")
    plt.ylabel("Test Accuracy")
    plt.xscale('log')
    plt.title("Tuning the Weight Decay")   
    '''
    plt.show()    
    
if __name__ == "__main__":
    main()    