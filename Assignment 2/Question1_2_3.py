import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random
np.random.seed(3)

with np.load("notMNIST.npz") as data:

    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]

    trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
    validData = validData.reshape((-1, 28*28)).astype(np.float32)
    testData = testData.reshape((-1, 28*28)).astype(np.float32)
    trainTarget = (np.arange(10) == trainTarget[:, None]).astype(np.float32)
    validTarget = (np.arange(10) == validTarget[:, None]).astype(np.float32)
    testTarget = (np.arange(10) == testTarget[:, None]).astype(np.float32)
    
    #Hyper Parameters
    learning_rate = 0.001
    num_epoch = 250
    minibatch_size = 500
    numTrainingSet = len(trainTarget)
    lam = 0.01
    
    #Graph
    x = tf.placeholder(tf.float32, [None, 784])
    y_target = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([10, 784]))
    b = tf.Variable(tf.zeros([1, 10]))

    y_pred = tf.add(tf.matmul(x, tf.transpose(W)), b)
    l_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_target, logits = y_pred))
    l_w = tf.scalar_mul(lam, tf.nn.l2_loss(W))
    cross_entropy_loss = l_d + l_w
    y_pred_softmax = tf.nn.softmax(logits = y_pred)
    #Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
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
        test_cost = sess.run(cross_entropy_loss, feed_dict={x: testData, y_target: testTarget})
        test_cross_entropy_loss.append(test_cost)
        # Test accuracy
        predictions = sess.run(y_pred_softmax, feed_dict={x: testData, y_target: testTarget})
        correct = 0.0
        count = 0
        for n in list(range(0, len(predictions))):
            if(np.argmax(predictions[n]) == np.argmax(testTarget[n])):
                correct += 1
            count += 1

        test_accuracy.append(correct / len(testTarget))

        #Training Cross Entropy Loss
        train_cost = sess.run(cross_entropy_loss, feed_dict={x: trainData, y_target: trainTarget})
        train_cross_entropy_loss.append(train_cost)
  
        #Training Accuracy
        predictions = sess.run(y_pred_softmax, feed_dict={x: trainData, y_target: trainTarget})
        correct = 0.0
        count = 0
        
        for n in list(range(0, len(predictions))):
            if(np.argmax(predictions[n]) == np.argmax(trainTarget[n])):
                correct += 1
            count += 1

        train_accuracy.append(correct / len(trainTarget))

    x = list(range(1, len(test_accuracy)+1))
    plt.plot(x, test_accuracy, label = 'Test Data' )
    plt.plot(x, train_accuracy, label='Training Data')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Epochs')
    plt.title('Multiclass Classifier Accuracy')
    plt.legend()
    plt.show()

    x = list(range(1, len(test_cross_entropy_loss) + 1))
    plt.plot(x, test_cross_entropy_loss, label='Test Data')
    plt.plot(x, train_cross_entropy_loss, label='Training Data')
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Number of Epochs')
    plt.title('Multiclass Classifier Cross Entropy')
    plt.legend()
    plt.show()
