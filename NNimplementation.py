class Neural_Network(object):
    def __init__(self, inputLayerSize = 784, outputLayerSize = 10, hiddenLayerSize1 = 10, hiddenLayerSize2 = 10):
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize1 = hiddenLayerSize1
        self.hiddenLayerSize2 = hiddenLayerSize2
        
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize1)
        self.W2 = np.random.randn(self.hiddenLayerSize1, self.hiddenLayerSize2)
        self.W3 = np.random.randn(self.hiddenLayerSize2, self.outputLayerSize)
    
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.W3)
        yHat = self.sigmoid(self.z4)
        return yHat
        
    def sigmoid(self, z):
        return 1/(1+np.e**(-z))
    
    def sigmoidPrime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def costfunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5 * np.sum((y - self.yHat)**2)
        return J, self.validate(y, self.yHat)
    
    def costfunctionprime(self, X, y):
        # Backprop last weight connecting hiddenLayer2 to output layer
        self.yHat = self.forward(X)
        dJdyHat = -(y-self.yHat)
        dyHatdZ4 = self.sigmoidPrime(self.z4)
        dZ4dW3 = self.a3
        delta4 = np.multiply(dJdyHat, dyHatdZ4)
        dJdW3 = np.dot(dZ4dW3.T, delta4)
        
        # Weight for hiddenLayer1 to hiddenLayer2
        dyHatdZ3 = self.sigmoidPrime(self.z3)
        dZ3dW2 = self.a2
        delta3 = np.multiply(np.dot(delta4, self.W3.T), dyHatdZ3)
        dJdW2 = np.dot(dZ3dW2.T, delta3)
        
        # Weight for inputLayer to hiddenLayer1
        dyHatdZ2 = self.sigmoidPrime(self.z2)
        delta2 = np.multiply(np.dot(delta3, self.W2.T), dyHatdZ2)
        dJdW1 = np.dot(X.T, delta2)
             
        return dJdW1, dJdW2, dJdW3
    
    # Compute the accuracy of prediction
    def validate(self, y, y_hat):
        y = vectorize_y(y)
        y_hat = vectorize_y(y_hat)
        diff = y - y_hat
        return (len(diff) - np.count_nonzero(diff)) * 1./len(diff)
    
    # Implement gradient descent
    def train(self, X, y, learning_rate = 0.1, num_iteration = 1000, step_size = 10):
        X_train, X_validate = X[:50000], X[50000:]
        Y_train, Y_validate = y[:50000], y[50000:]
           
        for _ in range(num_iteration):
            for i in range(0, X_train.shape[0], step_size):
                dJdW1, dJdW2, dJdW3 = self.costfunctionprime(
                    X_train[i:i+step_size], Y_train[i:i+step_size])
                self.W1 -= learning_rate * dJdW1
                self.W2 -= learning_rate * dJdW2
                self.W3 -= learning_rate * dJdW3

            train_cost, train_acc = self.costfunction(X_train, Y_train)
            val_cost, val_acc = self.costfunction(X_validate, Y_validate)
            print "[TRAIN] Cost: {0} Acc: {1} | [VAL] Cost: {2} Acc: {3}".format(train_cost, train_acc, val_cost, val_acc)    



        """
Methods related to working with digits data
"""

from scipy.io import loadmat
import numpy as np
from sklearn import preprocessing

TRAIN_DATA = "dataset/train.mat"
TEST_DATA = "dataset/test.mat"

def load_train_data():
    data = loadmat(TRAIN_DATA)
    train_data = get_samples_to_features(data)
    train_data = train_data.astype(float)
    train_data = preprocessing.normalize(train_data)
#     train_data = append_bias(train_data)
    train_labels = get_labels(data)
    train_labels = vectorize_labels(train_labels)
    train_data, train_labels = random_shuffle_data(train_data, train_labels)
    return train_data, train_labels

def load_test_data():
    test_data = loadmat(TEST_DATA)
    test_data = get_samples_to_features_from_train_images(test_data["test_images"], num_samples=10000)
    test_data = test_data.astype(float)
    test_data = preprocessing.normalize(test_data)
#     test_data = append_bias(test_data)
    return test_data

def random_shuffle_data(train_data, train_labels, print_shuffled_indices=True):
    shuffled_indices = np.arange(train_data.shape[0])
    np.random.shuffle(shuffled_indices)
    if print_shuffled_indices:
        print shuffled_indices
    return train_data[shuffled_indices], train_labels[shuffled_indices]

def get_samples_to_features_from_train_images(images, num_samples=60000):
    """
        Images is 28 x 28 x 60000 array
    """
    # 784 x 60000
    pixels_features_to_samples = np.reshape(images, (784, num_samples))
    # 60000 x 784
    samples_to_features = np.swapaxes(pixels_features_to_samples, 0, 1)
    return samples_to_features

def get_samples_to_features(train_data, key="train_images"):
    images = train_data[key]
    return get_samples_to_features_from_train_images(images)

def get_labels(train_data):
    return np.array([item[0] for item in train_data["train_labels"]])

def vectorize_labels(train_labels, num_output=10):
    a = np.zeros((train_labels.shape[0], num_output))
    for i in range(len(train_labels)):
        a[i][train_labels[i]] = 1
    return a

def vectorize_y(Y, num_output=10):
    return np.array([max(range(num_output), key=lambda x: sample[x]) for sample in Y])

def run_new(learning_rate=.1, hiddenLayerSize1=10, hiddenLayerSize2 = 10, num_iteration=1000, step_size=10):
    X, Y = load_train_data()
    n = Neural_Network()
    n.train(X, Y)

def run_existing(learning_rate=.1, hiddenLayerSize1=10, hiddenLayerSize2 = 10, num_iteration=1000, step_size=10):
    X, Y = load_train_data()
    n = load_existing_neural_network(hiddenLayerSize1=10, hiddenLayerSize2 = 10, num_iteration=1000, step_size=10)
    n.train(X, Y)

def predict_kaggle(hiddenLayerSize1=10, hiddenLayerSize2 = 10, num_iteration=1000, step_size=10):

    test_data = loadmat(TEST_DATA)
    test_data = get_samples_to_features_from_train_images(test_data["test_images"], num_samples=10000)
    test_data = test_data.astype(float)
    test_data = preprocessing.normalize(test_data)
    # X_test = append_bias(test_data)

    n = load_existing_neural_network(hidden_layer_size=hidden_layer_size, cross_entropy_loss=cross_entropy_loss)
    Y = n.forward(test_data)
    Y = vectorize_y(Y)
    with open('kaggle3', "w") as f:
        f.write("Id,Category\n")
        for i in range(1, len(Y)+1):
            f.write("{0},{1}\n".format(i, Y[i-1]))
 
        