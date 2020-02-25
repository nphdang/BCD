import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_olivetti_faces
import keras
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model, Sequential
from keras.datasets import mnist, cifar10
from keras import backend as K

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

np.random.seed(123)

### functions for VAE
# KL divergence layer
class KLDivergenceLayer(Layer):
    # identity transform layer that adds KL divergence to the final model loss
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)
    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

# Negative log likelihood (Bernoulli)
def nll(y_true, y_pred):
    # keras.losses.binary_crossentropy gives the mean over the last axis but we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

dataset = "olivetti" # mnist, cifar10, olivetti
print("dataset: {}".format(dataset))
### hyper-parameters of VAE and NN
if dataset == "mnist":
    # size of hidden layer (h)
    hidden_dim = 256
    # size of latent layer (z)
    latent_dim = 32 # 32
    batch_size = 256
    epochs = 50
if dataset == "cifar10":
    # size of hidden layer (h)
    hidden_dim = 512
    # size of latent layer (z)
    latent_dim = 32 # 32
    batch_size = 32
    epochs = 100
if dataset == "olivetti":
    # size of hidden layer (h)
    hidden_dim = 256
    # size of latent layer (z)
    latent_dim = 8 # 8
    batch_size = 32
    epochs = 200

# 1. read data
if dataset == "mnist":
    n_data = 60000
    (X_train_org, y_train_org), (X_test_org, y_test_org) = mnist.load_data()
    X_data, y_data = X_train_org[:n_data], y_train_org[:n_data]
    # get no of classes
    n_class = len(np.unique(y_data))
    # get data point size
    img_rows, img_cols = X_data.shape[1:]
    original_dim = img_rows * img_cols
if dataset == "cifar10":
    n_data = 50000
    (X_train_org, y_train_org), (X_test_org, y_test_org) = cifar10.load_data()
    X_data, y_data = X_train_org[:n_data], y_train_org[:n_data]
    # get no of classes
    n_class = len(np.unique(y_data))
    # get data point size
    img_rows, img_cols, img_chns = X_data.shape[1:]
    original_dim = img_rows * img_cols * img_chns
    input_shape = (img_rows, img_cols, img_chns)
if dataset == "olivetti":
    faces = fetch_olivetti_faces()
    X_data = faces.data
    X_image = faces.images
    y_data = faces.target
    # get no of classes
    n_class = len(np.unique(y_data))
    # get data point size
    img_rows, img_cols = X_image.shape[1:]
    original_dim = img_rows * img_cols

# 2. split data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2,
                                                    shuffle=True, stratify=y_data, random_state=42)
if dataset == "mnist":
    # normalize data
    X_train = np.expand_dims(X_train, axis=-1) / 255.
    X_test = np.expand_dims(X_test, axis=-1) / 255.
    # convert train and test sets from matrices to vectors
    X_train = X_train.reshape(-1, original_dim)
    X_test = X_test.reshape(-1, original_dim)
if dataset == "cifar10":
    X_train = X_train.reshape((X_train.shape[0], img_rows, img_cols, img_chns))
    X_test = X_test.reshape((X_test.shape[0], img_rows, img_cols, img_chns))
    # normalize data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
# convert labels from integers to one-hot encodings
y_train = keras.utils.to_categorical(y_train, n_class)
y_test = keras.utils.to_categorical(y_test, n_class)

# 3. train NN
# train NN once to obtain initial accuracy
time1 = timeit.default_timer()
if dataset == "mnist" or dataset == "olivetti":
    trained_model = Sequential()
    trained_model.add(Dense(hidden_dim, input_dim=original_dim, activation='relu'))
    trained_model.add(Dense(n_class, activation='softmax'))
if dataset == "cifar10":
    trained_model = Sequential()
    trained_model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                     activation='relu', input_shape=input_shape))
    trained_model.add(Conv2D(32, (3, 3), activation='relu'))
    trained_model.add(MaxPooling2D(pool_size=(2, 2)))
    trained_model.add(Dropout(0.25))
    trained_model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    trained_model.add(Conv2D(64, (3, 3), activation='relu'))
    trained_model.add(MaxPooling2D(pool_size=(2, 2)))
    trained_model.add(Dropout(0.25))
    trained_model.add(Flatten())
    trained_model.add(Dense(hidden_dim, activation='relu'))
    trained_model.add(Dropout(0.5))
    trained_model.add(Dense(n_class, activation='softmax'))
trained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model.fit(X_train, y_train, shuffle=True, epochs=epochs, batch_size=batch_size, verbose=0)
y_pred = trained_model.predict(X_test)
y_pred_round = np.around(y_pred)
accuracy = accuracy_score(y_test, y_pred_round)
print("accuracy={}".format(round(accuracy, 4)))
# save NN
trained_model.save("nn_{}.h5".format(dataset))
time2 = timeit.default_timer()
print("runtime of NN: {}(s)".format(round(time2 - time1, 2)))

# 4. train VAE
if dataset == "cifar10":
    # convert train and test sets from matrices to vectors
    X_train = X_train.reshape(-1, original_dim)
    X_test = X_test.reshape(-1, original_dim)
time1 = timeit.default_timer()
# Encoder: q(z|x)
x = Input(shape=(original_dim,))
h_q = Dense(hidden_dim, activation='relu')(x)
z_mu = Dense(latent_dim)(h_q)
z_log_var = Dense(latent_dim)(h_q)
z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
# equivalent to sample_z() in Keras
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
# z has a simple distribution N(0, 1)
eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_dim), mean=0.0, stddev=1.0))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])
encoder = Model(x, z_mu)
# Decoder: p(x|z)
decoder = Sequential([Dense(hidden_dim, input_dim=latent_dim, activation='relu'),
                      Dense(original_dim, activation='sigmoid')])
x_pred = decoder(z)
# train model
vae = Model(inputs=[x, eps], outputs=x_pred, name='vae')
vae.compile(optimizer='rmsprop', loss=nll)
hist = vae.fit(X_train, X_train, shuffle=True, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)
# save Encoder, Decoder, and VAE
encoder.save("encoder_{}.h5".format(dataset))
decoder.save("decoder_{}.h5".format(dataset))
vae.save("vae_{}.h5".format(dataset))
time2 = timeit.default_timer()
print("runtime of VAE: {}(s)".format(round(time2 - time1, 2)))
# evaluate model
golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))
fig, ax = plt.subplots(figsize=golden_size(6))
hist_df = pd.DataFrame(hist.history)
hist_df.plot(ax=ax)
ax.set_ylabel('loss')
ax.set_xlabel('# epochs')
ax.set_ylim(.99 * hist_df[1:].values.min(), 1.1 * hist_df[1:].values.max())
plt.savefig('./loss_vae_{}.pdf'.format(dataset), bbox_inches="tight")
plt.close()

