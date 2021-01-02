from cnn import classify_with_cnn
from mlp import classify_with_mlp
from utils import plot_learning_curve, get_data

print('Loading the dataset...')
mnist_data = get_data()
print('Dataset loaded.')
print('Training the MLP...')
mlp_training_history = classify_with_mlp(mnist_data)
print('MLP trained.')
print('Training the CNN model...')
cnn_training_history = classify_with_cnn(mnist_data)
print('CNN trained.')
print()
print('MLP Test Accuracy:', mlp_training_history['val_accuracy'][-1])
print('CNN Test Accuracy:', cnn_training_history['val_accuracy'][-1])
plot_learning_curve(mlp_training_history, 10)
plot_learning_curve(cnn_training_history, 10)
