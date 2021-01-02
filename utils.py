import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def plot_learning_curve(history, epochs):
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history['accuracy'])
    plt.plot(epoch_range, history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()

