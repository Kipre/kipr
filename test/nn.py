import kipr as kp
import numpy as np

with np.load("C:/Users/kipr/Downloads/mnist.npz") as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']

nb_examples = len(train_labels)

train = kp.arr((train_examples.reshape(-1, 28*28, 1)/255).astype(np.float32))

train_oh = np.zeros((nb_examples, 10))
np.put_along_axis(train_oh, train_labels.reshape(-1, 1), 1, 1)
labels = kp.arr(train_oh.astype(np.float32)).reshape([nb_examples, 10, 1])

drelu = lambda x: kp.relu(x) / (kp.relu(x) + kp.arr(0.000000000001))

epochs = 1
batch_size = 128
learning_rate = kp.arr(0.01)
nb_batches = nb_examples // batch_size

W1 = kp.arr('random', shape=[100, 784]) - kp.arr(0.5)
b = kp.arr('random', shape=[100, 1]) - kp.arr(0.5)
W2 = kp.arr('random', shape=[10, 100]) - kp.arr(0.5)

# h = W1 @ train[:batch_size] + b
# W2 @ kp.relu(h)

rmses = []

for k in range(epochs):
    for batch in range(nb_batches):
        x_batch = train[batch*batch_size : (batch+1)*batch_size]
        y_batch = labels[batch*batch_size : (batch+1)*batch_size]

        s1 = W1 @ x_batch
        s2 = s1 + b
        s3 = kp.relu(s2)
        s4 = W2 @ s3
        loss = (s4  - y_batch)
        rmse = (loss * loss) / kp.arr(2)
        rmses.append(rmse.mean())
        print(rmse.mean())

        dW2 = loss @ s3.reshape([batch_size, 1, 100])
        
        r1 = loss.reshape([batch_size, 1, 10]) @ W2
        r2 = r1.reshape([batch_size, 100, 1]) * drelu(s2)

        db = r2

        dW1 = r2 @ x_batch.reshape([batch_size, 1, 784]) 

        W1 -= learning_rate * dW1.mean(0)
        b -= learning_rate * b.mean(0)
        W2 -= learning_rate * dW2.mean(0)


