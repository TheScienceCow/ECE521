import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random
np.random.seed(3)

mode = 1 #0 for logistic, 1 for linear

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
    lam = 0
    
    #Graph
    x = tf.placeholder(tf.float32, [None, 784])
    y_target = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.zeros([1, 784]))
    b = tf.Variable(0.0)

    y_pred = tf.add(tf.matmul(x, tf.transpose(W)), b)
    if mode == 0:
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=y_pred))
    else:
      loss = tf.scalar_mul(0.5, tf.reduce_mean(tf.pow(tf.subtract(y_pred, y_target), 2)))
    y_pred_sigmoid = tf.nn.sigmoid(y_pred)
    #Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #Training
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)

      test_loss = []
      test_accuracy = []

      valid_loss = []
      valid_accuracy = []
      
      train_loss = []
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

        # Test Loss
        sum_cost = sess.run(loss, feed_dict={x: testData, y_target: testTarget})
        test_loss.append(sum_cost)
        # Test accuracy
        if mode == 0:
          predictions = sess.run(y_pred_sigmoid, feed_dict={x: testData, y_target: testTarget})
        else:
          predictions = sess.run(y_pred, feed_dict={x: testData, y_target: testTarget})
        correct = 0.0
        count = 0
        for n in list(range(0, len(predictions))):
            if predictions[n] > 0.5 and testTarget[n] > 0.5:
                correct += 1
            elif predictions[n] < 0.5 and testTarget[n] < 0.5:
                correct += 1
            count += 1

        test_accuracy.append(correct / len(testTarget))
        
        # Valid Loss
        sum_cost = sess.run(loss, feed_dict={x: validData, y_target: validTarget})
        valid_loss.append(sum_cost)
        # Valid accuracy
        if mode == 0:
          predictions = sess.run(y_pred_sigmoid, feed_dict={x: validData, y_target: validTarget})
        else:
          predictions = sess.run(y_pred, feed_dict={x: validData, y_target: validTarget})
        correct = 0.0
        count = 0
        for n in list(range(0, len(predictions))):
            if predictions[n] > 0.5 and validTarget[n] > 0.5:
                correct += 1
            elif predictions[n] < 0.5 and validTarget[n] < 0.5:
                correct += 1
            count += 1

        valid_accuracy.append(correct / len(validTarget))
        
        
        #Training Cross Entropy Loss
        sum_cost = sess.run(loss, feed_dict={x: trainData, y_target: trainTarget})
        train_loss.append(sum_cost)
  
        #Training Accuracy
        if mode == 0:
          predictions = sess.run(y_pred_sigmoid, feed_dict={x: trainData, y_target: trainTarget})
        else:
          predictions = sess.run(y_pred, feed_dict={x: trainData, y_target: trainTarget})
        correct = 0.0
        count = 0
        for n in list(range(0, len(predictions))):
            if predictions[n] > 0.5 and trainTarget[n] > 0.5:
                correct += 1
            elif predictions[n] < 0.5 and trainTarget[n] < 0.5:
                correct += 1
            count += 1

        train_accuracy.append(correct / len(trainTarget))

    x = list(range(1, len(test_loss) + 1))
    plt.plot(x, test_accuracy, label='Test accuracy')
    plt.plot(x, train_accuracy, label='Training accuracy')
    plt.plot(x, valid_accuracy, label='Validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Epochs')
    if mode == 0:
      plt.title('Logistic Regression')
    else:
      plt.title('Linear Regression')
    plt.legend()
    plt.show()
