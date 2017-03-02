import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import random

#hyperparameters for the programming :)
debug = False
num_labels = 10
learning_rate = np.exp(np.random.uniform(-7.5,-4.5,size=(5,1)))
weight_decay = np.exp(np.random.uniform(-9,-6,size=(5,1)))
trainingEpochs = 100
miniBatchSize = 500
report_period = 10
keepProbs = np.random.randint(1,3,size=(5,1))/2.0
f = open("./q2_5_1.log", "w")

def layerFunc(activation, numIn, numOut, id):
    '''
    layerFunc:
    accepts 4 inputs cause it's easier to work with ints. Also mimics the format given by Jimmy on Piazza
    activation - a 2-d tensor that has the dimensions [#examples,#activations] that represents the activation of the previous layerFunc
    numIn - the number of inputs, also the Dimension of a single activation (included for explicit description)
    numOut - the number of outputs.abs
    id - for referencing
    '''
    weights = tf.Variable(tf.truncated_normal([numIn, numOut], stddev=np.sqrt(3.0/(numIn+numOut)),dtype=tf.float64), name='weights{0}'.format(id))
    biases = tf.Variable(tf.zeros([numOut],dtype=tf.float64), name='biases{0}'.format(id))
    return tf.matmul(activation, weights) + biases, weights

def train(miniBatchSize,learningRate,weightDecay,trainingEpochs, keepProb):
    np.random.seed(random.randint(0,4294967295))
    print("Starting Training with parameters MBS {0}, LR {1}, WD {2}, TE {3}. Keep {4}".format(miniBatchSize,learningRate,weightDecay,trainingEpochs, keepProb), file = f)

    #building the neural network.
    print("Network Structure:", file=f)
    x = tf.placeholder(tf.float64,[None,28,28])
    x_reshaped = tf.reshape(x,[tf.shape(x)[0],784])
    print("input (784 nodes):", file=f)
    y_ = tf.placeholder(tf.float64,[None,10])
    kp = tf.placeholder(tf.float64,shape=())
    weight_array = []
    num_layers = np.random.random_integers(1,5)

    dim_last = 784
    activation_last = x_reshaped
    for i in range(num_layers): #drop-out between EVERY layer.
        num_HU = np.random.random_integers(100,500)
        print("Hidden layer {0}, with {1} HU".format(i,num_HU), file =f)
        z_h, h_weights = layerFunc(activation_last,dim_last,num_HU,i)
        weight_array.append(h_weights)
        h = tf.nn.relu(z_h)
        h_dropped = tf.nn.dropout(h, kp)
        activation_last = h_dropped
        dim_last = num_HU
    out, out_weights = layerFunc(h_dropped,dim_last,10,num_layers + 1)
    weight_array.append(out_weights)

    weights_l2 = tf.Variable(0, dtype=tf.float64)
    for weight in weight_array:
        weights_l2 += tf.nn.l2_loss(weight)
    
    #define our loss function, accuracy function, and training style
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=out)) + weight_decay*(weights_l2)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(out),1),tf.argmax(y_,1)),tf.float64))
    adam_train = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

    #load the dataset
    with np.load(os.path.join(os.path.dirname(__file__),"notMNIST.npz")) as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(1125)
        tf.set_random_seed(1127)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]

        #checking that the onehot works fine.
        if (debug):
            randIndx = np.arange(10)
            Data = Data[randIndx]
            Target = Target[randIndx]
        
        #let's convert to one-hot, thanks stack overflow
        NewTarget = np.eye(num_labels)[Target]
        trainData, trainTarget = Data[:15000], NewTarget[:15000]
        validData, validTarget = Data[15000:16000], NewTarget[15000:16000]
        testData, testTarget = Data[16000:], NewTarget[16000:]
        numTrainingSet = 15000

    #Tracking Lists
    updates = []
    validation_loss_list = []
    training_loss_list = []
    test_loss_list = []
    test_accuracy_list = []
    training_accuracy_list =[]
    validation_accuracy_list = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())
        
        # Fit all training data, minibatch style
        for epoch in range(trainingEpochs):
            randIndex = np.arange(numTrainingSet)
            np.random.shuffle(randIndex)
            shuffledData = trainData[randIndex]
            shuffledTarget = trainTarget[randIndex]
            start = 0
            for i in range(numTrainingSet//miniBatchSize):
                sess.run(adam_train, feed_dict={x: shuffledData[start:start+miniBatchSize], y_: shuffledTarget[start:start+miniBatchSize], kp: keepProb})
                start = start + miniBatchSize
            #Report if necessary 
            if epoch % report_period == 0:
                print("Epoch: ", epoch)
                print("Epoch: ", epoch, file = f)
                print("Cross_entropy: ", sess.run(cross_entropy, feed_dict={x: trainData, y_: trainTarget, kp: keepProb}), file = f)
                print("Accuracy: ", sess.run(accuracy, feed_dict={x: trainData, y_: trainTarget,kp: keepProb}), file=f)
        print ("Training Finished with Valid Accuracy as {0} and Valid Cross Entropy as {1},\n".format(sess.run(accuracy, feed_dict={x:validData, y_:validTarget,kp: 1}),sess.run(cross_entropy, feed_dict={x: validData, y_: validTarget,kp: 1})), file=f)
        print ("and with Test Accuracy as {0} and Test Cross Entropy as {1}\n".format(sess.run(accuracy, feed_dict={x:testData, y_:testTarget,kp: 1}),sess.run(cross_entropy, feed_dict={x: testData, y_: testTarget,kp: 1})), file=f)
    return updates, validation_loss_list, training_loss_list, test_loss_list, test_accuracy_list, training_accuracy_list, validation_accuracy_list

training_CE_all = []
training_acc_all = []
for i in range(len(keepProbs)):
    print ("Training set ", i + 1)
    u, vll, trll, tell, teal, tral, val = train(miniBatchSize, float(learning_rate[i][0]),float(weight_decay[i][0]),trainingEpochs,float(keepProbs[i][0]))
print("Finished Optimization")
f.close()