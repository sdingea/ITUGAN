import numpy as np
from keras.datasets import mnist
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip

def load_noise(n, noise_sz):
    random_state = np.random.RandomState(18)
    return np.array([random_state.uniform(-1, 1, noise_sz) for i in range(n)])

def load_mnist(n, preprocess = 0, matrix = np.zeros((28, 28))):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:n, :, :]
    if preprocess:
        for i in range(n):
            x_train[i] = np.dot(np.dot(matrix, x_train[i]), matrix.T)
    x_train = (((x_train).astype(np.float32) - 127.5) / 127.5)[:, np.newaxis, :, :]
    return x_train

def read_fashion_mnist(path):  
    with gzip.open(os.path.join(path, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        ret = np.frombuffer(f.read(), dtype = np.uint8, offset = 16).reshape(60000, 28, 28)
    return ret

def load_fashion_mnist(path, n, preprocess = 0, matrix = np.zeros((28, 28))):
    x_origin = read_fashion_mnist(path)
    x_origin = x_origin[:n, :, :]
    x_train = np.copy(x_origin) # 'x_origin' is read-only
    del(x_origin)
    if preprocess:
        for i in range(n):
            x_train[i] = np.dot(np.dot(matrix, x_train[i]), matrix.T)
    x_train = ((x_train.astype(np.float32) - 127.5) / 127.5)[:, np.newaxis, :, :]
    return x_train

def save_image(path, image, name, epoch):
    f, ax = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            ax[i, j].imshow(image[3 * i + j][0], interpolation = 'nearest', cmap = 'gray_r')
            ax[i, j].axis('off')
    f.set_size_inches(18.5, 10.5)
    if not os.path.exists(path):
        os.makedirs(path)
    f.savefig(path + name + '_after_' + str(epoch) + '_epoch.png', dpi = 100, bbox_inches = 'tight', pad_inches = 0)
    plt.close(f)
