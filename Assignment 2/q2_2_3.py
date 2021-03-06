import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


#hyperparameters for the programming :)
debug = False
num_labels = 10
learning_rates = [1e-3]
weight_decay = 3e-4
trainingEpochs = 100
miniBatchSize = 500
report_period = 1

f = open("./q2_2_3.log", "w")


def layerFunc(activation, numOut):
    '''
    layerFunc:
    activation - a 2-d tensor that has the dimensions [#examples,#activations] that represents the activation of the previous layerFunc
    numOut - the number of outputs.abs
    '''
    numIn = activation.get_shape().as_list()[1]
    weights = tf.Variable(tf.truncated_normal([numIn, numOut], stddev=3.0/(numIn+numOut),dtype=tf.float64))
    biases = tf.Variable(tf.zeros([numOut],dtype=tf.float64))
    return tf.matmul(activation, weights) + biases, weights

def train(miniBatchSize,learningRate,weightDecay,trainingEpochs):
    print("Starting Training with parameters MBS {0}, LR {1}, WD {2}, TE {3}".format(miniBatchSize,learningRate,weightDecay,trainingEpochs), file = f)
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
    x_reshaped = tf.reshape(x,[tf.shape(x)[0],784])
    y_ = tf.placeholder(tf.float64,[None,10])
    z_h, h_weights = layerFunc(x_reshaped,1000)
    h = tf.nn.relu(z_h)
    out, out_weights = layerFunc(h,10)

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
            updates.append(epoch+1)
            validation_loss_list.append(sess.run(cross_entropy, feed_dict={x: validData, y_: validTarget}))
            training_loss_list.append(sess.run(cross_entropy, feed_dict={x: trainData, y_: trainTarget}))
            test_loss_list.append(sess.run(cross_entropy, feed_dict={x: testData, y_: testTarget}))
            test_accuracy_list.append(sess.run(accuracy, feed_dict={x:testData, y_:testTarget}))
            training_accuracy_list.append(sess.run(accuracy, feed_dict={x:trainData, y_:trainTarget}))
            validation_accuracy_list.append(sess.run(accuracy, feed_dict={x:validData, y_:validTarget}))
            if epoch % report_period == 0:
                print("Epoch: ", epoch)
                print("Epoch: ", epoch, file = f)
                print("Cross_entropy: ", training_loss_list[-1], file = f)
                print("Accuracy: ", training_accuracy_list[-1], file=f)
                
                print("V Cross_entropy: ", validation_loss_list[-1], file = f)
                print("V Accuracy: ", validation_accuracy_list[-1], file=f)

                
                print("Te Cross_entropy: ", test_loss_list[-1], file = f)
                print("Te Accuracy: ", test_accuracy_list[-1], file=f)
            if (epoch > 1):
                if validation_loss_list[-1] > validation_loss_list[-2]:
                    return updates, validation_loss_list, training_loss_list, test_loss_list, test_accuracy_list, training_accuracy_list, validation_accuracy_list
        print ("Training Finished with Test Accuracy as {0} and Test Cross Entropy as {1}\n".format(sess.run(accuracy, feed_dict={x:testData, y_:testTarget}),sess.run(cross_entropy, feed_dict={x: testData, y_: testTarget})), file=f)
    return updates, validation_loss_list, training_loss_list, test_loss_list, test_accuracy_list, training_accuracy_list, validation_accuracy_list

training_CE_all = []
training_acc_all = []
validation_CE_all = []
validation_acc_all = []
test_CE_all = []
test_acc_all = []
for i in range(len(learning_rates)):
    print ("Training set ", i + 1)
    u, vll, trll, tell, teal, tral, val = train(miniBatchSize, learning_rates[i],weight_decay,trainingEpochs)
    plt.figure()
    plt.plot(u,trll, label="Training CE")
    plt.plot(u,vll, label="Validation CE")
    plt.plot(u,tell, label="Test CE")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-entropy")
    plt.title("Cross-entropy vs. Epochs, for learning rate = {0}".format(learning_rates[i]))
    plt.legend()
    plt.figure()
    plt.plot(u,tral, label="Training")
    plt.plot(u,val, label="Validation")
    plt.plot(u,teal, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs for learning rate = {0}".format(learning_rates[i]))
    plt.legend()
    plt.figure()
    plt.plot(u,1-np.array(tral), label="Training")
    plt.plot(u,1-np.array(val), label="Validation")
    plt.plot(u,1-np.array(teal), label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error vs. Epochs for learning rate = {0}".format(learning_rates[i]))
    plt.legend()
    training_CE_all.append(trll)
    training_acc_all.append(tral)
'''
plt.figure()

for i in range(len(learning_rates)):
    plt.plot(np.arange(trainingEpochs) + 1, training_CE_all[i], label="rate = {0}".format(learning_rates[i]))
plt.xlabel('Epochs')
plt.ylabel('Training Cross Entropy')
plt.title("Training Cross Entropy for several learning rates")
plt.legend()
plt.figure()
for i in range(len(learning_rates)):
    plt.plot(np.arange(trainingEpochs) + 1, 1- np.array(training_acc_all[i]), label="rate = {0}".format(learning_rates[i]))
plt.xlabel('Epochs')
plt.ylabel('Training Classification Error')
plt.title("Training Classification Error for several learning rates")
plt.legend()
'''
f.close()
plt.show()
print("Finished Optimization")