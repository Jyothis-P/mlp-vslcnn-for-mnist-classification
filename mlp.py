import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# hyper-parameters
learning_rate = 0.01
epochs = 10
batch_size = 32
step_display = 1

n_input = 784
n_hidden_1 = 128
n_hidden_2 = 32
n_output = 10


# format dataset
def to_one_hot(y):
    data = np.zeros(n_output)
    data[y] = 1
    return data


# defining accuracy function
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    acc = (100.0 * correctly_predicted) / predictions.shape[0]
    return acc


# creating the model
def multilayer_perceptron(x, parameters):
    l0 = x
    l1 = tf.nn.sigmoid(tf.matmul(l0, parameters['w1']) + parameters['b1'])
    l2 = tf.nn.sigmoid(tf.matmul(l1, parameters['w2']) + parameters['b2'])
    l3 = tf.matmul(l2, parameters['w3']) + parameters['b3']
    return l3


def classify_with_mlp(data):
    (x_train, y_train), (x_test, y_test) = data

    x_train = np.reshape(x_train, (-1, n_input))
    x_test = np.reshape(x_test, (-1, n_input))

    y_train = np.array([to_one_hot(y) for y in y_train])
    y_test = np.array([to_one_hot(y) for y in y_test])

    print('Shapes:', x_train.shape, y_train.shape)

    # model placeholders
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])

    # parameters of the model
    params = {'w1': tf.Variable(tf.random_uniform([n_input, n_hidden_1], -1.0, 1.0)),
              'w2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], -1.0, 1.0)),
              'w3': tf.Variable(tf.random_uniform([n_hidden_2, n_output], -1.0, 1.0)),
              'b1': tf.Variable(tf.zeros([n_hidden_1])), 'b2': tf.Variable(tf.zeros([n_hidden_2])),
              'b3': tf.Variable(tf.zeros([n_output]))}

    # construct the model
    logits = multilayer_perceptron(x, params)
    pred = tf.nn.softmax(logits)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # initiating the variables
    init = tf.global_variables_initializer()

    train_accuracy = []
    test_accuracy = []

    # training model
    with tf.Session() as sess:
        # run initializer
        sess.run(init)

        # training cycle
        for epoch in range(epochs):
            avg_cost = 0
            avg_acc = 0
            total_batch = len(x_train) // batch_size

            for i in range(total_batch):
                batch_x = x_train[i:i + 1 * batch_size]
                batch_y = y_train[i:i + 1 * batch_size]

                _, c = sess.run([train_op, loss_op], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch

                pred_y = sess.run(logits, feed_dict={x: batch_x})
                acc = accuracy(pred_y, batch_y)
                avg_acc += acc / total_batch

            if (epoch + 1) % step_display == 0:
                test_loss = sess.run(loss_op, feed_dict={x: x_test, y: y_test})
                pred_y = sess.run(pred, feed_dict={x: x_test})
                test_acc = accuracy(pred_y, y_test)
                train_accuracy.append(avg_acc)
                test_accuracy.append(test_acc)
                print("Epoch: {:2.0f} - Loss: {:1.5f} - Acc: {:2.5f} - Test Loss: {:1.5f} - Test Acc: {:2.5f}".format(
                    epoch + 1, avg_cost, avg_acc, test_loss, test_acc))

        print("Optimization finished")

        return {'accuracy': train_accuracy, 'val_accuracy': test_accuracy}
