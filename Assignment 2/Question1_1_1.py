import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random
np.random.seed(3)

with np.load("notMNIST.npz") as data :
    #Setup
    Data, Target = data ["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target == posClass) + (Target == negClass)
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target == posClass] = 1
    Target[Target == negClass] = 0
    randIndx = np.arange(len(Data))
    rng.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]

    trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
    validData = validData.reshape((-1, 28*28)).astype(np.float32)
    testData = testData.reshape((-1, 28*28)).astype(np.float32)
    
    
    
    #Hyper Parameters
    learning_rate = 0.001
    num_epoch = 250
    minibatch_size = 500
    numTrainingSet = len(trainTarget)
    lam = 0.01
    
    #Graph
    x = tf.placeholder(tf.float32, [None, 784])
    y_target = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.zeros([1, 784]))
    b = tf.Variable(0.0)

    y_pred = tf.add(tf.matmul(x, tf.transpose(W)), b)
    l_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=y_pred))
    l_w = tf.scalar_mul(lam, tf.nn.l2_loss(W))
    cross_entropy_loss = l_d + l_w
    y_pred_sigmoid = tf.nn.sigmoid(y_pred)
    #Gradient Descent Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)
    
    #Training
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)

      test_cross_entropy_loss = []
      test_accuracy = []

      train_cross_entropy_loss = []
      train_accuracy = []
      
      #SGD
      for epoch in range(num_epoch):
        randIndex = np.arange(numTrainingSet)
        rng.shuffle(randIndex)
        shuffledData = trainData[randIndex]
        shuffledTarget = trainTarget[randIndex]
        start = 0
        for i in range(numTrainingSet//minibatch_size):
          sess.run(optimizer, feed_dict={x: shuffledData[start:start+minibatch_size], y_target: shuffledTarget[start:start+minibatch_size]})
          start = start + minibatch_size

        # Test Cross Entropy Loss
        sum_cost = sess.run(cross_entropy_loss, feed_dict={x: testData, y_target: testTarget})
        test_cross_entropy_loss.append(sum_cost)
        # Test accuracy
        predictions = sess.run(y_pred_sigmoid, feed_dict={x: testData, y_target: testTarget})
        correct = 0.0
        count = 0
        for n in list(range(0, len(predictions))):
            if predictions[n] > 0.5 and testTarget[n] > 0.5:
                correct += 1
            elif predictions[n] < 0.5 and testTarget[n] < 0.5:
                correct += 1
            count += 1

        test_accuracy.append(correct / len(testTarget))

        #Training Cross Entropy Loss
        sum_cost = sess.run(cross_entropy_loss, feed_dict={x: trainData, y_target: trainTarget})
        train_cross_entropy_loss.append(sum_cost)
  
        #Training Accuracy
        predictions = sess.run(y_pred_sigmoid, feed_dict={x: trainData, y_target: trainTarget})
        correct = 0.0
        count = 0
        for n in list(range(0, len(predictions))):
            if predictions[n] > 0.5 and trainTarget[n] > 0.5:
                correct += 1
            elif predictions[n] < 0.5 and trainTarget[n] < 0.5:
                correct += 1
            count += 1

        train_accuracy.append(correct / len(trainTarget))

    x = list(range(1, len(test_accuracy)+1))
    plt.plot(x, test_accuracy, label = 'Test Data' )
    plt.plot(x, train_accuracy, label='Training Data')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Epochs')
    plt.title('Logistic Regression Accuracy with LR = 0.001')
    plt.legend()
    plt.show()

    x = list(range(1, len(test_cross_entropy_loss) + 1))
    plt.plot(x, test_cross_entropy_loss, label='Test Data')
    plt.plot(x, train_cross_entropy_loss, label='Training Data')
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Number of Epochs')
    plt.title('Logistic Regression Cross Entropy Loss with LR = 0.001')
    plt.legend()
    plt.show()
