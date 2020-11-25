import kipr as kp
import numpy as np
import matplotlib.pyplot as plt

with np.load("C:/Users/kipr/Downloads/mnist.npz") as data:
    train_examples = (data['x_train'].reshape(-1, 28*28, 1)/255).astype(np.float32)
    train_labels = np.zeros((60000, 10))
    np.put_along_axis(train_labels, data['y_train'].reshape(-1, 1), 1, 1)
    train_labels = train_labels.astype(np.float32).reshape(-1, 10, 1)
    test_examples = (data['x_test'].reshape(-1, 28*28, 1)/255).astype(np.float32)
    test_labels = np.zeros((10000, 10))
    np.put_along_axis(test_labels, data['y_test'].reshape(-1, 1), 1, 1)
    test_labels = test_labels.astype(np.float32).reshape(-1, 10, 1)



nb_examples = len(train_labels)

train = kp.arr(train_examples)
labels = kp.arr(train_labels)

class LinNN:

    learning_rate = kp.arr(0.5)
    
    def __init__(self, i, h, o):
        self.i = i
        self.h = h
        self.o = o
        self.W1 = kp.arr('rand', shape=[h, i]) - kp.arr(0.5)
        self.b = kp.arr('rand', shape=[h, 1]) - kp.arr(0.5)
        self.W2 = kp.arr('rand', shape=[o, h]) - kp.arr(0.5)

    def step(self, x_batch, y_batch):
        
        drelu = lambda x: kp.relu(x) / (kp.relu(x) + kp.arr(0.000000000001))

        s1 = self.W1 @ x_batch
        s2 = s1 + self.b
        s3 = kp.relu(s2)
        s4 = self.W2 @ s3
        s5 = kp.softmax(s4.reshape([self.batch_size, self.o])).reshape([self.batch_size, self.o, 1])
        loss = (s5 - y_batch)
        x_entropy = (-y_batch*kp.log(s5)).mean()

        dW2 = loss @ s3.reshape([self.batch_size, 1, self.h])
        
        r1 = loss.reshape([self.batch_size, 1, self.o]) @ self.W2
        r2 = r1.reshape([self.batch_size, self.h, 1]) * drelu(s2)

        db = r2

        dW1 = r2 @ x_batch.reshape([self.batch_size, 1, 784]) 

        self.W1 -= self.learning_rate * dW1.mean(0)
        self.b -= self.learning_rate * db.mean(0)
        self.W2 -= self.learning_rate * dW2.mean(0)
        return x_entropy.mean()
    
    def train(self, train, labels, epochs=1, batch_size=64, verbose=20):
        self.batch_size = batch_size
        for k in range(epochs):
            for batch in range(train.shape[0]//batch_size):
                x_batch = train[batch*batch_size : (batch+1)*batch_size]
                y_batch = labels[batch*batch_size : (batch+1)*batch_size]

                if batch % verbose == 0:
                    print(self.step(x_batch, y_batch))

    def accuracy(self, x_test, y_test):

        predictions = kp.softmax(self.W2 @ kp.relu(self.W1 @ x_test.reshape([-1, self.i, 1]) + self.b), -2)

        success = predictions.numpy().argmax(-2) == y_test.argmax(-2)
        return success.mean()
    

nn = LinNN(784, 32, 10)

nn.train(train, labels, epochs=10)

print(nn.accuracy(kp.arr(test_examples), test_labels))

