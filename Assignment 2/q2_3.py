import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#hyperparameters for the programming :)
debug = False
num_labels = 10
learning_rate = 1e-3
weight_decay = 3e-4
trainingEpochs = 100
miniBatchSize = 500
report_period = 10
numHiddenUnits = [100, 500, 1000]

f = open("./q2_3.log", "w")

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

def train(miniBatchSize,learningRate,weightDecay,trainingEpochs,numHiddenUnits):
    print("Starting Training with parameters MBS {0}, LR {1}, WD {2}, TE {3}, NHU {4}".format(miniBatchSize,learningRate,weightDecay,trainingEpochs,numHiddenUnits), file = f)
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
    z_h, h_weights = layerFunc(x_reshaped,784,numHiddenUnits,0)
    h = tf.nn.relu(z_h)
    out, out_weights = layerFunc(h,numHiddenUnits,10,1)

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
                sess.run(adam_train, feed_dict={x: shuffledData[start:start+miniBatchSize], y_: shuffledTarget[start:start+miniBatchSize]})
                start = start + miniBatchSize
            #Report if necessary 
            if epoch % report_period == 0:
                print("Epoch: ",epoch)
                print("Epoch: ", epoch, file=f)
                print("Cross_entropy: ", sess.run(cross_entropy, feed_dict={x: trainData, y_: trainTarget}), file = f)
                print("Accuracy: ", sess.run(accuracy, feed_dict={x: trainData, y_: trainTarget}), file = f)
            updates.append(epoch+1)
            validation_loss_list.append(sess.run(cross_entropy, feed_dict={x: validData, y_: validTarget}))
            training_loss_list.append(sess.run(cross_entropy, feed_dict={x: trainData, y_: trainTarget}))
            test_loss_list.append(sess.run(cross_entropy, feed_dict={x: testData, y_: testTarget}))
            test_accuracy_list.append(sess.run(accuracy, feed_dict={x:testData, y_:testTarget}))
            training_accuracy_list.append(sess.run(accuracy, feed_dict={x:trainData, y_:trainTarget}))
            validation_accuracy_list.append(sess.run(accuracy, feed_dict={x:validData, y_:validTarget}))
        print ("Training Finished with Test Accuracy as {0} and Test Cross Entropy as {1}\n".format(sess.run(accuracy, feed_dict={x:testData, y_:testTarget}),sess.run(cross_entropy, feed_dict={x: testData, y_: testTarget})), file=f)   
    return updates, validation_loss_list, training_loss_list, test_loss_list, test_accuracy_list, training_accuracy_list, validation_accuracy_list

training_CE_all = []
training_acc_all = []
validation_CE_all = []
validation_acc_all = []
for i in range(len(numHiddenUnits)):
    print ("Training set", i + 1)
    u, vll, trll, tell, teal, tral, val = train(miniBatchSize, learning_rate,weight_decay,trainingEpochs,numHiddenUnits[i])
    plt.figure()
    plt.plot(u,trll, label="Training CE")
    plt.plot(u,vll, label="Validation CE")
    plt.plot(u,tell, label="Test CE")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-entropy")
    plt.title("Cross-entropy vs. Epochs, for {0} hidden units".format(numHiddenUnits[i]))
    plt.legend()
    plt.figure()
    plt.plot(u,tral, label="Training")
    plt.plot(u,val, label="Validation")
    plt.plot(u,teal, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs for {0} hidden units".format(numHiddenUnits[i]))
    plt.legend()
    plt.figure()
    plt.plot(u,1-np.array(tral), label="Training")
    plt.plot(u,1-np.array(val), label="Validation")
    plt.plot(u,1-np.array(teal), label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error vs. Epochs for {0} hidden units    ".format(numHiddenUnits[i]))
    plt.legend()
    training_CE_all.append(trll)
    training_acc_all.append(tral)
    validation_CE_all.append(vll)
    validation_acc_all.append(val)
    test_CE_all.append(vll)
    test_acc_all.append(val)
plt.figure()
for i in range(len(numHiddenUnits)):
    plt.plot(np.arange(trainingEpochs) + 1, training_CE_all[i], label="{0} hidden units".format(numHiddenUnits[i]))
plt.xlabel('Epochs')
plt.ylabel('Training Cross Entropy')
plt.title("Training Cross Entropy for variation in hidden units")
plt.legend()
plt.figure()
for i in range(len(numHiddenUnits)):
    plt.plot(np.arange(trainingEpochs) + 1, 1- np.array(training_acc_all[i]), label="{0} hidden units".format(numHiddenUnits[i]))
plt.xlabel('Epochs')
plt.ylabel('Training Classification Error')
plt.title("Training Classification Error for variation in hidden units")
plt.legend()
plt.figure()
for i in range(len(numHiddenUnits)):
    plt.plot(np.arange(trainingEpochs) + 1, 1- np.array(validation_CE_all[i]), label="{0} hidden units".format(numHiddenUnits[i]))
plt.xlabel('Epochs')
plt.ylabel('Validation Cross Entropy')
plt.title("Validation Cross Entropy for variation in hidden units")
plt.legend()
plt.figure()
for i in range(len(numHiddenUnits)):
    plt.plot(np.arange(trainingEpochs) + 1, 1- np.array(validation_acc_all[i]), label="{0} hidden units".format(numHiddenUnits[i]))
plt.xlabel('Epochs')
plt.ylabel('Validation Classification Error')
plt.title("Validation Classification Error for variation in hidden units")
plt.legend()
plt.show()
print("Finished Optimization")
