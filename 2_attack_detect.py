import numpy as np
import copy
import timeit
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_olivetti_faces
import keras
from keras.datasets import mnist, cifar10
from keras import backend as K
from keras.models import load_model
from keras.layers import Layer
from numpy import linalg as LA
from scipy.optimize import minimize
import multiprocessing
import GPy
import GPyOpt

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

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

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
    batch_size = 256
    epochs = 100
if dataset == "olivetti":
    # size of hidden layer (h)
    hidden_dim = 256
    # size of latent layer (z)
    latent_dim = 8 # 8
    batch_size = 32
    epochs = 200

# read data
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

# split data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2,
                                                    shuffle=True, stratify=y_data, random_state=42)
n_train = len(y_train)
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

# load trained models: NN, Encoder, Decoder, and VAE
nn = load_model("nn_{}.h5".format(dataset))
nn_modified = copy.deepcopy(nn)
vae = load_model("vae_{}.h5".format(dataset),
                 custom_objects={'latent_dim': latent_dim, 'KLDivergenceLayer': KLDivergenceLayer, 'nll': nll})
encoder = load_model("encoder_{}.h5".format(dataset),
                     custom_objects={'latent_dim': latent_dim, 'KLDivergenceLayer': KLDivergenceLayer, 'nll': nll})
decoder = load_model("decoder_{}.h5".format(dataset))

# check accuracy of original model
y_pred = nn.predict(X_test)
y_pred_round = np.around(y_pred)
accuracy = accuracy_score(y_test, y_pred_round)
print("original accuracy={}".format(round(accuracy, 4)))

# get NN weights
first_layer_weights = nn.layers[0].get_weights()[0]
first_layer_biases = nn.layers[0].get_weights()[1]
second_layer_weights = nn.layers[1].get_weights()[0]
second_layer_biases = nn.layers[1].get_weights()[1]

# functions to compute sensitivity score
def relu(x):

    return np.where(x > 0, x, 0.0)

def softmax(z):
    exps = np.exp(z)

    return exps / np.sum(exps)

# derivative of softmax
def dersoftmax(z):
    lenm = z.shape[0]
    arsum = sum(z)
    for i in range(0, lenm):
        z[i] = z[i] * (arsum - z[i]) / (arsum * arsum)
    return z

def final_der(x, y):
    d = np.zeros((hidden_dim, 0))
    lenar = y.shape[0]
    for il in range(0, lenar):
        d = np.concatenate((d, np.array(x * y[il]).reshape(hidden_dim, 1)), axis=1)

    return d

# objective func of BO optimization
def obj_func_bo(z):
    z = z.reshape(-1, latent_dim)
    x_sample = decoder.predict(z)
    x_sample = x_sample.reshape(-1, original_dim)
    x_sample = x_sample.ravel()
    a1 = x_sample.dot(first_layer_weights) + first_layer_biases
    h1 = relu(a1)
    a2 = h1.dot(second_layer_weights) + second_layer_biases
    y = softmax(a2)
    dersoft_y = dersoftmax(y)
    der_matrix = final_der(h1, dersoft_y)
    outnorm = LA.norm(der_matrix)

    return outnorm

# objective function of local optimization
def obj_func_local(x):
    x = x.reshape(-1, original_dim)
    x = x.ravel()
    a1 = x.dot(first_layer_weights) + first_layer_biases
    h1 = relu(a1)
    a2 = h1.dot(second_layer_weights) + second_layer_biases
    y = softmax(a2)
    dersoft_y = dersoftmax(y)
    der_matrix = final_der(h1, dersoft_y)
    outnorm = LA.norm(der_matrix)

    return outnorm

# BO hyper-parameters
n_iter = 60
n_dim = latent_dim
n_init = n_dim + 1
# bounds for BO optimization
z_bound = [-3.0, 3.0]
bounds_bo = [{"name": "z{}".format(d+1), "type": "continuous", "domain": (z_bound[0], z_bound[1])}
             for d in range(n_dim)]

# bounds for local optimization
x_bound = [0.0, 1.0] # x is normalized
bounds_local = np.zeros((original_dim, 2))
for d in range(original_dim):
    bounds_local[d, :] = np.array([x_bound[0], x_bound[1]])

n_runs = 10
n_modify_model = 500
rate_rnd_run_ratio, rate_local_run_ratio, rate_bo_run_ratio = \
    np.zeros((n_runs, 9)), np.zeros((n_runs, 9)), np.zeros((n_runs, 9))
for run in range(n_runs):
    print("run: {}".format(run))
    np.random.seed(run)
    # get a random image
    rnd = np.random.randint(0, n_train)
    x0 = X_train[rnd]
    x0 = x0.reshape(-1, original_dim)
    plt.axis('off')
    if dataset == "olivetti":
        plt.imshow(x0.reshape(img_rows, img_cols), cmap='Greys_r')
    else:
        plt.imshow(x0.reshape(img_rows, img_cols))
    plt.savefig("org_image_{}_{}.pdf".format(dataset, run), bbox_inches="tight")
    plt.close()
    '''local optimization'''
    def minus_func(x):
        return -obj_func_local(x)
    v = x0
    local_opt = minimize(fun=minus_func, x0=v, method='L-BFGS-B', bounds=bounds_local)
    sensitive_img_local = local_opt.x
    sensitive_img_local = sensitive_img_local.reshape(-1, original_dim)
    plt.axis('off')
    if dataset == "olivetti":
        plt.imshow(sensitive_img_local.reshape(img_rows, img_cols), cmap='Greys_r')
    else:
        plt.imshow(sensitive_img_local.reshape(img_rows, img_cols))
    plt.savefig("sensitive_image_local_{}_{}.pdf".format(dataset, run), bbox_inches="tight")
    plt.close()
    '''BO optimization'''
    img = x0
    x0_encoded = encoder.predict(img)
    # initialize points for BO
    np.random.seed(run)
    data_bo = np.zeros((n_init, n_dim))
    for i in range(n_init):
        z_init = np.array([np.random.uniform(bounds_bo[d]["domain"][0], bounds_bo[d]["domain"][1], 1)[0]
                           for d in range(n_dim)])
        data_bo[i, :] = z_init
    # the first initial point is x0_encoded
    data_bo[0, :] = x0_encoded
    result_bo = []
    for z_init in data_bo:
        y_init = obj_func_bo(z_init)
        result_bo.append(y_init)
    # reformat result_bo
    result_bo = np.array(result_bo).reshape(-1, 1)
    # run BO to find sensitive sample
    for t in range(n_iter):
        print("iteration: {}".format(t))
        my_kernel = GPy.kern.ExpQuad(input_dim=latent_dim, variance=1.0, lengthscale=1.0)
        bo = GPyOpt.methods.BayesianOptimization(f=None, num_cores=multiprocessing.cpu_count(),
                                                 domain=bounds_bo,
                                                 X=data_bo, Y=result_bo,
                                                 normalize_Y=True, maximize=True,
                                                 kernel=my_kernel,
                                                 acquisition_type="LCB", acquisition_weight=1.0)
        # suggest x_next
        x_next = bo.suggest_next_locations()
        # compute y_next
        y_next = obj_func_bo(x_next)
        y_next = np.array(y_next).reshape(-1, 1)
        # augment data
        data_bo = np.vstack((data_bo, x_next))
        result_bo = np.vstack((result_bo, y_next))
    # get bestx and besty
    besty = np.max(result_bo)
    bestx = data_bo[np.argmax(result_bo)]
    print("sensitive_image_bo: {}, sensitivity_score: {}".format(bestx, besty))
    # get sensitive image
    bestx = bestx.reshape(-1, latent_dim)
    sensitive_img_bo = decoder.predict(bestx)
    plt.axis('off')
    if dataset == "olivetti":
        plt.imshow(sensitive_img_bo.reshape(img_rows, img_cols), cmap='Greys_r')
    else:
        plt.imshow(sensitive_img_bo.reshape(img_rows, img_cols))
    plt.savefig("sensitive_image_bo_{}_{}.pdf".format(dataset, run), bbox_inches="tight")
    plt.close()

    rate_rnd_all, rate_local_all, rate_bo_all = [], [], []
    for ratio_idx in range(0, 9):
        if ratio_idx == 0:
            ratio_keep = 0.01
        else:
            ratio_keep = ratio_idx * 0.1
        print("ratio_keep: {}".format(ratio_keep))

        # use sensitive image to detect if the model was modified
        rate_rnd, rate_local, rate_bo = 0, 0, 0
        l2 = []
        for modify_idx in range(n_modify_model):
            print("modify_idx: {}".format(modify_idx))
            np.random.seed(modify_idx)
            # modify original model to generate compromised model
            noise2b = np.random.normal(0, .01, n_class)
            indices = np.random.choice(range(noise2b.size), replace=False,
                                       size=int(noise2b.size * ratio_keep))
            noise2b[indices] = 0
            noise2w = np.array([np.array(noise2b), ] * hidden_dim)
            l2.append(second_layer_weights + noise2w)
            l2.append(second_layer_biases + noise2b)
            nn_modified.layers[1].set_weights(l2)

            # check accuracy of modified model
            y_pred = nn_modified.predict(X_test)
            y_pred_round = np.around(y_pred)
            accuracy = accuracy_score(y_test, y_pred_round)
            print("modified accuracy={}".format(round(accuracy, 4)))

            # random method
            pred_org = nn.predict(x0)
            class_org = np.argmax(pred_org, axis=1)[0]
            pred_modify = nn_modified.predict(x0)
            class_modify = np.argmax(pred_modify, axis=1)[0]
            if class_org != class_modify:
                rate_rnd = rate_rnd + 1
            # local method
            pred_org = nn.predict(sensitive_img_local)
            class_org = np.argmax(pred_org, axis=1)[0]
            pred_modify = nn_modified.predict(sensitive_img_local)
            class_modify = np.argmax(pred_modify, axis=1)[0]
            if class_org != class_modify:
                rate_local = rate_local + 1
            # bo method
            pred_org = nn.predict(sensitive_img_bo)
            class_org = np.argmax(pred_org, axis=1)[0]
            pred_modify = nn_modified.predict(sensitive_img_bo)
            class_modify = np.argmax(pred_modify, axis=1)[0]
            if class_org != class_modify:
                rate_bo = rate_bo + 1

            # clear the last layer
            l2 = []

        rate_rnd = rate_rnd / n_modify_model
        rate_local = rate_local / n_modify_model
        rate_bo = rate_bo / n_modify_model
        print("detection rate, random: {}, local: {}, bo: {}".
              format(round(rate_rnd, 4), round(rate_local, 4), round(rate_bo, 4)))
        rate_rnd_all.append(rate_rnd)
        rate_local_all.append(rate_local)
        rate_bo_all.append(rate_bo)

    rate_rnd_run_ratio[run, :] = rate_rnd_all
    rate_local_run_ratio[run, :] = rate_local_all
    rate_bo_run_ratio[run, :] = rate_bo_all

# save results to file
with open("rate_rnd_run_ratio_{}.file".format(dataset), "wb") as f:
    np.save(f, rate_rnd_run_ratio)
with open("rate_local_run_ratio_{}.file".format(dataset), "wb") as f:
    np.save(f, rate_local_run_ratio)
with open("rate_bo_run_ratio_{}.file".format(dataset), "wb") as f:
    np.save(f, rate_bo_run_ratio)

print("dataset: {}".format(dataset))
print("random method: {}".format(rate_rnd_run_ratio.mean(axis=0)))
print("local method: {}".format(rate_local_run_ratio.mean(axis=0)))
print("bo method: {}".format(rate_bo_run_ratio.mean(axis=0)))

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(end_time-start_time, 2)))

