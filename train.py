import sys
import numpy as np
from sklearn.utils import shuffle
from keras import backend as backend
from iolib import load_mnist, load_fashion_mnist, load_noise, save_image
from matrixlib import get_blur_matrix

N_SAMPLE = 20000
BATCH_SZ = 200
N_BATCH = N_SAMPLE / BATCH_SZ
N_EPOCH = 200
DECAY_THRESHOLD = 100
LR = 0.0002
EDGE = 28
NOISE_SZ = 100 
BLUR_MATRIX = get_blur_matrix(0.1, EDGE)
FASHION_MNIST_PATH = '../data/fashion'

def main():
	'''
	initialize data & save one original picture
	'''
	x_train = load_fashion_mnist(FASHION_MNIST_PATH, N_SAMPLE)
	x_noise = load_noise(N_SAMPLE, NOISE_SZ)
	x_origin = np.copy(x_train)
	for i in range(12):
		save_image('images/samples/', x_origin, 'source', -1 - i)
		x_origin = x_origin[12:]
	del(x_origin)
	
	'''
	construct the model
	'''
	from keras.models import Sequential
	from keras.models import Model
	from keras.layers import Dense
	from keras.layers.core import Activation
	from keras.layers import Reshape
	from keras.layers.convolutional import UpSampling2D
	from keras.layers.convolutional import Convolution2D
	from keras.layers.advanced_activations import LeakyReLU
	from keras.layers.core import Flatten
	from keras.layers import Input
	from keras.layers.core import Lambda
	from keras.optimizers import Adam
	from keras import initializations

	adam = Adam(lr = 0.0002, beta_1 = 0.5)

	def init_normal(shape, name = None, dim_ordering = None):
		return initializations.normal(shape, scale = 0.02, name = name)

	g = Sequential()
	g.add(Dense(input_dim = NOISE_SZ, output_dim = (128 * (EDGE / 4) * (EDGE / 4)), init = init_normal))	
	g.add(Activation('relu'))
	g.add(Reshape((128, (EDGE / 4), (EDGE / 4))))
	g.add(UpSampling2D(size = (2, 2)))
	g.add(Convolution2D(64, 5, 5, border_mode = 'same'))
	g.add(Activation('relu'))
	g.add(UpSampling2D(size = (2, 2)))
	g.add(Convolution2D(1, 5, 5, border_mode = 'same'))
	g.add(Activation('tanh'))
	g.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

	d = Sequential()
	d.add(Convolution2D(64, 5, 5, border_mode = 'same', subsample = (2, 2), input_shape = (1, EDGE, EDGE), init = init_normal))
	d.add(LeakyReLU(0.2))
	d.add(Convolution2D(128, 5, 5, border_mode = 'same', subsample = (2, 2)))
	d.add(LeakyReLU(0.2))
	d.add(Flatten())
	d.add(Dense(1))
	d.add(Activation('sigmoid'))
	d.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

	def t_model(x):
		m_ts = backend.variable(value = np.array([BLUR_MATRIX for i in range(BATCH_SZ)], dtype = 'f'))
		m_ts_t = backend.variable(value = np.array([BLUR_MATRIX.T for i in range(BATCH_SZ)], dtype = 'f'))	
		x = backend.squeeze(x, 1)
		x = backend.batch_dot(backend.batch_dot(m_ts, x), m_ts_t)
		x = backend.expand_dims(x, dim = 1)
		return x

	d.trainable = False
	dcgan_input = Input(shape = (NOISE_SZ, ))
	dcgan_output = d(Lambda(t_model, output_shape = lambda input_shape : input_shape)(g(dcgan_input)))
	dcgan = Model(input = dcgan_input, output = dcgan_output)
	dcgan.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
	d.trainable = True

	
	'''
	train the model
	'''
	def t(x):
		tx = np.copy(x)
		for i in range(BATCH_SZ):
			tx[i][0] = np.dot(np.dot(BLUR_MATRIX, tx[i][0]), BLUR_MATRIX.T)
		return tx
	for epoch in range(1, N_EPOCH + 1):
		print
		print('Epoch: ', epoch)
		for i in range(N_BATCH):
			print '.',
			sys.stdout.flush()
			nois_bat = x_noise[np.random.randint(N_SAMPLE, size = BATCH_SZ)]
			orig_bat = x_train[np.random.randint(N_SAMPLE, size = BATCH_SZ)]
			g_bat = g.predict(nois_bat)
			tr_bat = np.concatenate((t(g_bat), orig_bat), axis = 0)
			tr_label = np.concatenate((np.zeros(BATCH_SZ).astype(int), np.ones(BATCH_SZ).astype(int)))
			tr_bat, tr_label = shuffle(tr_bat, tr_label)
			d.train_on_batch(tr_bat, tr_label)	#train the discriminator
			d.trainable = False
			dcgan.train_on_batch(nois_bat, np.ones(BATCH_SZ).astype(int))	#train the generator
			d.trainable = True
		if (epoch % 5 == 0) or (epoch == 1):
			save_image('images/generated/', g_bat, 'generated', epoch)
		if epoch > DECAY_THRESHOLD:
			d.optimizer.lr.set_value((d.optimizer.lr.get_value() - LR / DECAY_THRESHOLD).astype(np.float32))
			g.optimizer.lr.set_value((g.optimizer.lr.get_value() - LR / DECAY_THRESHOLD).astype(np.float32))
	print('Complete')

main()
