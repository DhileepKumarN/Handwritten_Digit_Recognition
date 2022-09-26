#Import necessary packages
import numpy as np
from matplotlib import pyplot as plt
import cv2
#Load the Dataset and Normalize
from sklearn import datasets
from sklearn.model_selection import train_test_split
mnist = datasets.load_digits()
X = mnist.data
y = mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)
X_train_new = np.zeros((X_train.shape[0], 1024))
X_test_new = np.zeros((X_test.shape[0], 1024))
for i in range(len(X_train)):
    temp = np.reshape(X_train[i], (8, 8))
    X_train_new[i] = cv2.resize(temp, (32, 32)).flatten()
for i in range(len(X_test)):
    temp = np.reshape(X_test[i], (8, 8))
    X_test_new[i] = cv2.resize(temp, (32, 32)).flatten()
mean = np.mean(X_train_new, axis = 0)
std = np.std(X_train_new, axis = 0)
X_train = (X_train_new - mean)/(std + 1e-8)
mean = np.mean(X_test_new, axis = 0)
std = np.std(X_test_new, axis = 0)
X_test = (X_test_new - mean)/(std + 1e-8)
print(len(X_train))

#Build and Train LeNet 5 model from scratch

#Function to Perform tanh activation
def tanh(x):
    x = np.maximum(x, -50)
    return (1 - np.exp(-2 * x))/(1 + np.exp(-2 * x))

#Function to Calculate Softmax Activation
def softmax(x):
    x = x - np.max(x)
    return np.exp(x)/(np.sum(np.exp(x)) + 1e-8)

#Function to Perform Convolution with valid padding
def convolve_valid(inp, ker):
    if inp.ndim == 2 and ker.ndim == 2:
        W1 = inp.shape[0]
        F = ker.shape[0]
        W2 = W1 - F + 1
        result = np.zeros((W2, W2))
        for i in range(W2):
            for j in range(W2):
                for k in range(F):
                    for l in range(F):
                        result[i][j] = result[i][j] + ker[k][l] * inp[k + i][l + j]
        return result
    if inp.ndim == 2:
        W1 = inp.shape[0]
        F = ker.shape[1]
        W2 = W1 - F + 1
        D = ker.shape[0]
        result = np.zeros((D, W2, W2))
        for i in range(D):
            for j in range(W2):
                for k in range(W2):
                    for l in range(F):
                        for m in range(F):
                            result[i][j][k] = result[i][j][k] + ker[i][l][m] * inp[j + l][k + m]
    else:
        W1 = inp.shape[1]
        F = ker.shape[2]
        W2 = W1 - F + 1
        D = ker.shape[0]
        result = np.zeros((D, W2, W2))
        for i in range(D):
            for j in range(W2):
                for k in range(W2):
                    for l in range(inp.shape[0]):
                        for m in range(F):
                            for n in range(F):
                                result[i][j][k] = result[i][j][k] + ker[i][l][m][n] * inp[l][j + m][k + n]
    return result

#Function to perform average pooling
def avg_pool(inp, pool_size = 2, stride = 2):
    n = inp.shape[0]
    result_size = (int)((inp.shape[1] - pool_size)/stride + 1)
    result = np.zeros((n, result_size, result_size))
    for i in range(n):
        for j in range(result_size):
            for k in range(result_size):
                result[i][j][k] = np.mean(inp[i][j * pool_size: (j + 1) * pool_size][k * pool_size: (k + 1) * pool_size])
    return result

#Function to Perform Forward Propagation
def forward_prop_Q2(convW1, convW2, W1, W2, W3, b1, b2, b3, x_i):
    conva1 = convolve_valid(x_i, convW1)
    convy1 = tanh(conva1)
    pooly1 = avg_pool(convy1)
    conva2 = convolve_valid(pooly1, convW2)
    convy2 = tanh(conva2)
    pooly2 = avg_pool(convy2)
    fcx = pooly2.flatten()
    fcx = np.reshape(fcx, (-1, 1))
    a1 = np.matmul(W1, fcx) + b1
    y1 = tanh(a1)
    a2 = np.matmul(W2, y1) + b2
    y2 = tanh(a2)
    a3 = np.matmul(W3, y2) + b3
    y_pred = softmax(a3)
    return y_pred, y2, y1, fcx, convy2,  pooly1, convy1

#Function to Perform Backprop of Average Pooling Layer
def avgpool_backprop(inp, pool_size = 2):
    rsize = inp.shape[1] * 2
    result = np.zeros((inp.shape[0], rsize, rsize))
    for i in range(result.shape[0]):
        for j in range(rsize):
            for k in range(rsize):
                result[i][j][k] = inp[i][(int)(j/2)][(int)(k/2)]/(pool_size * pool_size)
    return result

#Function to Calculate Gradients wrt Weights
def convolve_backprop_weights(dconv, pool):
    if pool.ndim > 2:
        result = np.zeros((dconv.shape[0], pool.shape[0], -dconv.shape[1] + pool.shape[1] + 1, -dconv.shape[2] + pool.shape[2] + 1))
        for i in range(pool.shape[0]):
            result[:, i, :, :] = convolve_valid(pool[i], dconv)
        return result
    else:
        result = np.zeros((dconv.shape[0], -dconv.shape[1] + pool.shape[0] + 1, -dconv.shape[2] + pool.shape[1] + 1))
        for i in range(dconv.shape[0]):
            result[i] = convolve_valid(pool, dconv[i])
        return result

#Function to Calculate gradients wrt Inputs
def convolve_backprop_inputs(dconv, convW):
    convW = np.rot90(convW, axes = (2, 3), k = 2)
    dconv = np.pad(dconv, pad_width = ((0, 0), (4, 4), (4, 4)))
    result = np.zeros((convW.shape[1], dconv.shape[1] - convW.shape[2] + 1, dconv.shape[2] - convW.shape[3] + 1))
    for i in range(result.shape[0]):
        result = result + convolve_valid(dconv[i], convW[i])
    return result

#Function to Calculate Gradients
def calc_gradients_Q2(y_pred, y, y2, W3, y1, W2, fcx, W1, convy2,  pooly1, convW2, convy1, x):
    y = np.reshape(y, (-1, 1))
    delta_3 = (y_pred - y)
    db3 = delta_3
    dW3 = np.matmul(delta_3, y2.T)
    delta_2 = np.matmul(W3.T, delta_3) * (1 - y2 ** 2)
    db2 = delta_2
    dW2 = np.matmul(delta_2, y1.T)
    delta_1 = np.matmul(W2.T, delta_2) * (1 - y1 ** 2)
    db1= delta_1
    dW1 = np.matmul(delta_1, fcx.T)
    dfcx = np.matmul(W1.T, delta_1) * (1 - fcx ** 2)
    dfcx = np.reshape(dfcx, (-1, 5, 5))
    dconvy2 = avgpool_backprop(dfcx)
    dconva2 = dconvy2 * (1 - convy2 ** 2)
    dconvW2 = convolve_backprop_weights(dconva2, pooly1)
    dpooly1 = convolve_backprop_inputs(dconva2, convW2)
    dconvy1 = avgpool_backprop(dpooly1)
    dconva1 = dconvy1 * (1 - convy1 ** 2)
    dconvW1 = convolve_backprop_weights(dconva1, x)
    return dW3, dW2, dW1, db3, db2, db1, dconvW2, dconvW1

#Function to Calculate Error
def CCE_Q2(X_train, y_train, convW1, convW2, W1, W2, W3, b1, b2, b3):
    y_predicted = np.zeros((X_train.shape[0], 10))
    for i in range(len(X_train)):
        x_i = X_train[i]
        conva1 = convolve_valid(x_i, convW1)
        convy1 = tanh(conva1)
        pooly1 = avg_pool(convy1)
        conva2 = convolve_valid(pooly1, convW2)
        convy2 = tanh(conva2)
        pooly2 = avg_pool(convy2)
        fcx = pooly2.flatten()
        fcx = np.reshape(fcx, (-1, 1))
        a1 = np.matmul(W1, fcx) + b1
        y1 = tanh(a1)
        a2 = np.matmul(W2, y1) + b2
        y2 = tanh(a2)
        a3 = np.matmul(W3, y2) + b3
        y_pred = softmax(a3)
        y_predicted[i] = y_pred.flatten()
    y_train = np.eye(10)[np.array(y_train, dtype = int)]
    cost = -1 * np.mean(np.sum(y_train * np.log10(y_predicted + 1e-8), axis = 1))
    return cost
    
#Function to Perform Stochastic Gradient Descent
def SGD_Q2(convW1, convW2, W1, W2, W3, b1, b2, b3, X_train, y_train, num_epochs = 50, eta = 0.01, verbose = True):
    cost_arr = np.zeros(num_epochs)
    for i in range(num_epochs):
        k = 1
        for x_i, y_i in zip(X_train, y_train):
            print(k,)
            k = k + 1
            y_i = np.eye(10)[(int)(y_i)]
            #Forward Propagation
            y_pred, y2, y1, fcx, convy2,  pooly1, convy1 = forward_prop_Q2(convW1, convW2, W1, W2, W3, b1, b2, b3, x_i)
            #Calculate Gradients
            dW3, dW2, dW1, db3, db2, db1, dconvW2, dconvW1 = calc_gradients_Q2(y_pred, y_i, y2, W3, y1, W2, fcx, W1, convy2,  pooly1, convW2, convy1, x_i)
            #Backpropagation
            W3 = W3 - eta * dW3
            W2 = W2 - eta * dW2
            W1 = W1 - eta * dW1
            b3 = b3 - eta * db3
            b2 = b2 - eta * db2
            b1 = b1 - eta * db1
            convW2 = convW2 - eta * dconvW2
            convW1 = convW1 - eta * dconvW1
        #Calculate Loss
        cost_arr[i] = CCE_Q2(X_train, y_train, convW1, convW2, W1, W2, W3, b1, b2, b3)
        if (verbose):
            print("Epoch Number : %i \t CCE = %10f"%(i+1, cost_arr[i]))
    return cost_arr, convW1, convW2, W1, W2, W3, b1, b2, b3

#Initialize Weights
convW1 = np.random.randn(6, 5, 5) * 0.1
convW2 = np.random.randn(16, 6, 5, 5) * 0.1
W1 = np.random.randn(120, 400) * 0.1
W2 = np.random.randn(84, 120) * 0.1
W3 = np.random.randn(10, 84) * 0.1
b1 = np.random.randn(120, 1) * 0.1
b2 = np.random.randn(84, 1) * 0.1
b3 = np.random.randn(10, 1) * 0.1
X_train = np.reshape(X_train, (-1, 32, 32))
X_train = X_train[0:200]
y_train = y_train[0:200]
cost_arr, convW1, convW2, W1, W2, W3, b1, b2, b3 = SGD_Q2(convW1, convW2, W1, W2, W3, b1, b2, b3, X_train, y_train)

'''
#Dry Run
conva1 = convolve_valid(X_train[0], convW1)
convy1 = tanh(conva1)
pooly1 = avg_pool(convy1)
conva2 = convolve_valid(pooly1, convW2)
convy2 = tanh(conva2)
pooly2 = avg_pool(convy2)
fcx = pooly2.flatten()
fcx = np.reshape(fcx, (-1, 1))
a1 = np.matmul(W1, fcx) + b1
y1 = tanh(a1)
a2 = np.matmul(W2, y1) + b2
y2 = tanh(a2)
a3 = np.matmul(W3, y2) + b3
y_pred = softmax(a3)
y = np.eye(10)[(int)(y_train[0])]
y = np.reshape(y, (-1, 1))
delta_3 = (y_pred - y)
dW3 = np.matmul(delta_3, y2.T)
delta_2 = np.matmul(W3.T, delta_3) * (1 - y2 ** 2)
dW2 = np.matmul(delta_2, y1.T)
delta_1 = np.matmul(W2.T, delta_2) * (1 - y1 ** 2)
dW1 = np.matmul(delta_1, fcx.T)
dfcx = np.matmul(W1.T, delta_1) * (1 - fcx ** 2)
dfcx = np.reshape(dfcx, (-1, 5, 5))
dconvy2 = avgpool_backprop(dfcx)
dconva2 = dconvy2 * (1 - convy2 ** 2)
dconvW2 = convolve_backprop_weights(dconva2, pooly1)
dpooly1 = convolve_backprop_inputs(dconva2, convW2)
dconvy1 = avgpool_backprop(dpooly1)
dconva1 = dconvy1 * (1 - convy1 ** 2)
dconvW1 = convolve_backprop_weights(dconva1, X_train[0])
dconvW1.shape
'''

#Plot Learning Curve
plt.figure(figsize = (15, 7))
epoch_index = [(i + 1) for i in range(50)]
plt.plot(epoch_index, cost_arr)
plt.xlabel('Epoch Index')
plt.ylabel('BCE Value')
plt.title('Learning Curve')
plt.show()

#Build LeNET 5 Model using Tensorflow
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

model_Q2 = Sequential()
model_Q2.add(Conv2D(filters = 6, kernel_size = 5, activation = 'tanh', input_shape = (32, 32, 1)))
model_Q2.add(AveragePooling2D(pool_size = (2, 2), strides = (2, 2)))
model_Q2.add(Conv2D(filters = 16, kernel_size = 5, activation = 'tanh'))
model_Q2.add(AveragePooling2D(pool_size = (2, 2), strides = (2, 2)))
model_Q2.add(Flatten())
model_Q2.add(Dense(units = 120, activation = 'tanh'))
model_Q2.add(Dense(units = 84, activation = 'tanh'))
model_Q2.add(Dense(units = 10, activation = 'softmax'))

model_Q2.compile(optimizer = 'SGD', loss = 'categorical_crossentropy')
model_Q2.summary()

#Train the Model Using the Training Data
y_train = np.eye(10)[np.array(y_train, dtype = int)]
X_train = np.reshape(X_train, (-1, 32, 32))
history = model_Q2.fit(X_train, y_train, epochs = 50)
cost_arr = history.history['loss']

plt.figure(figsize = (15, 7))
epoch_index = [(i + 1) for i in range(50)]
plt.plot(epoch_index, cost_arr)
plt.xlabel('Epoch Index')
plt.ylabel('MSE Value')
plt.title('Learning Curve')
plt.show()



#Test the Model
X_test = np.reshape(X_test, (-1, 32, 32))
y_pred = model_Q2.predict(X_test)
y_pred_act = np.argmax(y_pred, axis = 1)
for i in range(5):
    temp = X_test_new[i]
    temp = np.reshape(temp, (32, 32))
    plt.figure()
    plt.imshow(temp, cmap = 'gray')
    plt.title('Actual : %d, Predicted : %d'%(y_test[i], y_pred_act[i]))
    plt.show()
