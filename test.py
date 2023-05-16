import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
import model as nbModel
import nb_loss as nbLoss

import print_functions as pf


# initial values 
scale_factor= 1
size= 10_000
poisson_lambda = 1

pv_orig = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
r_orig_fakedata = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] 
p_orig_fakedata = [0.7, 0.3]
y_nb = np.zeros(size)
simulated_events = np.zeros((size, 10))

# Hyperparameters 
r_size= simulated_events.shape[1]
p_size = 2

batchSize= 1000
epochSize= 700
learningRateOne = 0.0018


# create fake data --> sample eventx from a poisson distribution, and y from negative binomial distributions 

for count, i in enumerate(pv_orig):
    c = (np.random.poisson(poisson_lambda, size) + 1).astype('float32')    
    y_nb +=  (np.random.negative_binomial(c * r_orig_fakedata[count], p_orig_fakedata[i])).astype('float32')
    simulated_events[:, count] = c
    
    #print(mean(r_orig_fakedata[count], p_orig_fakedata[i]))
    #print(var(r_orig_fakedata[count], p_orig_fakedata[i]))

y_nb = y_nb.reshape(list(y_nb.shape)+[1])

# This function creates the model and calls it

def run_model(matchdata, injury_time, batchSize, epochSize, learningRate, partition_vector = None): 

    nbl = nbModel.NegativeBinomialLayer(r_size, p_size, partition_vector)
    model = keras.Sequential([nbl])
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate), loss = nbLoss.my_loss_fn)
    history = model.fit(matchdata, injury_time, batch_size=batchSize, epochs=epochSize, verbose = True)

    r = nbl.get_r().numpy()
    p = nbl.get_p().numpy()
    pv = nbl.get_pv1().numpy()
    
    return r, p, pv, nbl


# Call model with the generated data 

r_train, p_train, pv_fuzzy_train, nbl = run_model(simulated_events, y_nb, batchSize, epochSize, learningRateOne)

pv_train = np.log(pv_fuzzy_train / (1-pv_fuzzy_train)) 

r_train, p_train, pv_train, nbl = run_model(simulated_events, y_nb, batchSize, epochSize, learningRateOne, pv_train)


# print result 

pf.print_results(simulated_events, y_nb, scale_factor, nbl, p_orig_fakedata, r_orig_fakedata, pv_orig)
