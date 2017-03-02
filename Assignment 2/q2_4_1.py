import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

#hyperparameters for the programming :)
debug = False
num_labels = 10
learning_rate = 1e-3
weight_decay = 0#3e-4
trainingEpochs = 200
miniBatchSize = 500
report_period = 10
keepProbs = [0.5, 1]
f = open("./q2_4_1.log", "w")

def layerFunc(activation, numIn, numOut, id):
    '''
    layerFunc:
    accepts 4 inputs cause it's easier to work with ints. Also mimics the format given by Jimmy on Piazza
    activation - a 2-d tensor that has the dimensions [#examples,#activations] that represents the activation of the previous layerFunc
    numIn - the number of inputs, also the Dimension of a single activation (included for explicit description)
    numOut - the number of outputs.abs
    id - for referencing
    '''
    weights = tf.Variable(tf.truncated_normal([numIn, numOut], stddev=3.0/(numIn+numOut),dtype=tf.float64), name='weights{0}'.format(id))
    biases = tf.Variable(tf.zeros([numOut],dtype=tf.float64), name='biases{0}'.format(id))
    return tf.matmul(activation, weights) + biases, weights

def train(miniBatchSize,learningRate,weightDecay,trainingEpochs, keepProb):
    print("Starting Training with parameters MBS {0}, LR {1}, WD {2}, TE {3}. Keep {4}".format(miniBatchSize,learningRate,weightDecay,trainingEpochs, keepProb), file = f)
    #load the dataset
    with np.load(os.path.join(os.path.dirname(__file__),"notMNIST.npz")) as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        tf.set_random_seed(1125)
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

    #building the neural network.
    x = tf.placeholder(tf.float64,[None,28,28])
    x_reshaped = tf.reshape(x,[tf.shape(x)[0],-1])
    y_ = tf.placeholder(tf.float64,[None,10])
    kp = tf.placeholder(tf.float64,shape=())
    z_h, h_weights = layerFunc(x_reshaped,784,1000,0)
    h = tf.nn.relu(z_h)
    h_dropped = tf.nn.dropout(h, kp)
    out, out_weights = layerFunc(h_dropped,1000,10,1)
    
    #define our loss function, accuracy function, and training style
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=out) + weight_decay*(tf.nn.l2_loss(h_weights) + tf.nn.l2_loss(out_weights)))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(out),1),tf.argmax(y_,1)),tf.float64))
    adam_train = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

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
            updates.append(epoch+1)
            validation_loss_list.append(sess.run(cross_entropy, feed_dict={x: validData, y_: validTarget,kp: 1}))
            training_loss_list.append(sess.run(cross_entropy, feed_dict={x: trainData, y_: trainTarget,kp: keepProb}))
            test_loss_list.append(sess.run(cross_entropy, feed_dict={x: testData, y_: testTarget,kp: 1}))
            test_accuracy_list.append(sess.run(accuracy, feed_dict={x:testData, y_:testTarget,kp: 1}))
            training_accuracy_list.append(sess.run(accuracy, feed_dict={x:trainData, y_:trainTarget,kp: keepProb}))
            validation_accuracy_list.append(sess.run(accuracy, feed_dict={x:validData, y_:validTarget,kp: 1}))
        print ("Training Finished with Test Accuracy as {0} and Test Cross Entropy as {1}\n".format(sess.run(accuracy, feed_dict={x:testData, y_:testTarget,kp: 1}),sess.run(cross_entropy, feed_dict={x: testData, y_: testTarget,kp: 1})), file=f)
    return updates, validation_loss_list, training_loss_list, test_loss_list, test_accuracy_list, training_accuracy_list, validation_accuracy_list

training_CE_all = []
training_acc_all = []
for i in range(len(keepProbs)):
    print ("Training set ", i + 1)
    u, vll, trll, tell, teal, tral, val = train(miniBatchSize, learning_rate,weight_decay,trainingEpochs,keepProbs[i])
    plt.figure()
    plt.plot(u,trll, label="Training CE")
    plt.plot(u,vll, label="Validation CE")
    plt.plot(u,tell, label="Test CE")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-entropy")
    plt.title("Cross-entropy vs. Epochs, for keep_prob = {0}".format(keepProbs[i]))
    plt.legend()
    plt.figure()
    plt.plot(u,tral, label="Training")
    plt.plot(u,val, label="Validation")
    plt.plot(u,teal, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs for keep_prob = {0}".format(keepProbs[i]))
    plt.legend()
    plt.figure()
    plt.plot(u,(1-np.array(tral))*2724, label="Training")
    plt.plot(u,(1-np.array(val))*2724, label="Validation")
    #plt.plot(u,1-np.array(teal), label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Number of Errors")
    plt.title("Error vs. Epochs for keep_prob = {0}".format(keepProbs[i]))
    plt.legend()
    training_CE_all.append(trll)
    training_acc_all.append(tral)
plt.figure()
for i in range(len(keepProbs)):
    plt.plot(np.arange(trainingEpochs) + 1, training_CE_all[i], label="keepProbs = {0}".format(keepProbs[i]))
plt.xlabel('Epochs')
plt.ylabel('Training Cross Entropy')
plt.title("Training Cross Entropy with and without Dropout")
plt.legend()
plt.figure()
for i in range(len(keepProbs)):
    plt.plot(np.arange(trainingEpochs) + 1, 1- np.array(training_acc_all[i]), label="keepProbs = {0}".format(keepProbs[i]))
plt.xlabel('Epochs')
plt.ylabel('Training Classification Error')
plt.title("Training Classification Error with and without Dropout")
plt.legend()
f.close()
plt.show()
print("Finished Optimization")
