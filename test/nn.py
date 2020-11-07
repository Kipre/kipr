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
        self.W1 = kp.arr('random', shape=[h, i]) - kp.arr(0.5)
        self.b = kp.arr('random', shape=[h, 1]) - kp.arr(0.5)
        self.W2 = kp.arr('random', shape=[o, h]) - kp.arr(0.5)

    def __call__(self, x):
        return self.W2 @ kp.relu(self.W1 @ x.reshape([self.i, 1]) + self.b)

    def step(self, x_batch, y_batch):
        
        drelu = lambda x: kp.relu(x) / (kp.relu(x) + kp.arr(0.000000000001))

        s1 = self.W1 @ x_batch
        s2 = s1 + self.b
        s3 = kp.relu(s2)
        s4 = self.W2 @ s3
        s5 = kp.softmax(s4.reshape([self.batch_size, self.o])).reshape([self.batch_size, self.o, 1])
        loss = (s5 - y_batch)
        x_entropy = (-y_batch*kp.ln(s5)).mean()

        dW2 = loss @ s3.reshape([self.batch_size, 1, self.h])
        
        r1 = loss.reshape([self.batch_size, 1, self.o]) @ self.W2
        r2 = r1.reshape([self.batch_size, self.h, 1]) * drelu(s2)

        db = r2

        dW1 = r2 @ x_batch.reshape([self.batch_size, 1, 784]) 

        self.W1 -= self.learning_rate * dW1.mean(0)
        self.b -= self.learning_rate * db.mean(0)
        self.W2 -= self.learning_rate * dW2.mean(0)
        return x_entropy.mean().val()
    
    def train(self, train, labels, epochs=1, batch_size=64, verbose=20):
        self.batch_size = batch_size
        for k in range(epochs):
            for batch in range(train.shape[0]//batch_size):
                x_batch = train[batch*batch_size : (batch+1)*batch_size]
                y_batch = labels[batch*batch_size : (batch+1)*batch_size]

                if batch % verbose == 0:
                    print(self.step(x_batch, y_batch))
    

nn = LinNN(784, 32, 10)

nn.train(train, labels)





# class npNN:

#     learning_rate = 0.01

#     def __init__(self, i, h, o):
#         self.i = i
#         self.h = h
#         self.o = o
#         self.W1 = np.random.randn(h, i)
#         self.b = np.random.randn(h, 1)
#         self.W2 = np.random.randn(o, h)

#     def __call__(self, x):
#         a = self.W1 @ x + self.b
#         b = self.W2 @ (a * (a > 0))
#         return np.exp(b)/np.exp(b).sum(1)[..., np.newaxis]
    
#     def step(self, x_batch, y_batch):

#         softmax = lambda x: np.exp(x)/np.exp(x).sum(1)[..., np.newaxis]

#         f1 = self.W1 @ x_batch                                               # (batch, h, 1) =  (h, i)        x (b, i, 1)
#         f2 = f1 + self.b                                                     # (batch, h, 1) =  (batch, h, 1) + (h, 1)
#         f3 = f2 * (f2 > 0)                                                   # (batch, h, 1) =  (batch, h, 1)
#         # print(f3)
#         f4 = self.W2 @ f3                                                    # (batch, o, 1) =  (o, h)        x (batch, h, 1)
#         f5 = softmax(f4)
#         # print(f5)
#         f6 = f5 - y_batch                                                    # (batch, o, 1) =  (batch, o, 1) - (batch, o, 1)
#         loss = -(y_batch * np.log(f5)).sum(-1)
#         rmse = loss.mean()

#         b1 = f6                                                              # (batch, o, 1) =  (batch, o, 1)
#         dW2 = b1 @ f3.reshape(-1, 1, self.h)                                 # (batch, o, h) =  (batch, o, 1) x (batch, 1, h)
#         b2 = (b1.reshape(-1, 1, self.o) @ self.W2).reshape(-1, self.h, 1)    # (batch, h, 1) = ((batch, 1, o) x (o, h)).reshape(batch, h, 1)
#         b3 = b2 * (f2 > 0)                                                   # (batch, h, 1) =  (batch, h, 1)
#         db = b3
#         dW1 = b3 @ x_batch.reshape(-1, 1, self.i)                            # (batch, h, i) =  (batch, h, 1) x (batch, i, 1)

#         # print(dW2, db, dW1)

#         self.W1 -= self.learning_rate * dW1.mean(0)
#         self.b -= self.learning_rate * db.mean(0)
#         self.W2 -= self.learning_rate * dW2.mean(0)

#         return rmse

#     def fit(self, x_data, y_data, epochs=3, batch_size=64, verbose=20):
#             for epoch in range(epochs):
#                 for batch in range(len(x_data) // batch_size):
                    
#                     x_batch = x_data[batch*batch_size:(batch+1)*batch_size]
#                     y_batch = y_data[batch*batch_size:(batch+1)*batch_size]

#                     loss = self.step(x_batch, y_batch)
#                     if batch % verbose == 0:
#                         print(loss)

#     def accuracy(self, x_test, y_test):

#         predictions = self.__call__(x_test)

#         success = predictions.argmax(-2) == y_test.argmax(-2)
#         return success.mean()



    
# npnn = npNN(784, 64, 10)

# npnn.fit(train_examples, train_labels,
#          epochs=10, verbose=100)
