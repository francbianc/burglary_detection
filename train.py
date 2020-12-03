import os
from os import listdir
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adagrad
from scipy.io import savemat

import configuration as cfg 
import classifier as clf
from loss_function import *
from load_trainset import *


# PATHS: cfg.path_all_features, cfg.trained_folder
# VARIABLES: cfg.train_exp_name, cfg.use_i3d, cfg.use_lstm, cfg.num_0, cfg.num_1
# AIM: training the classifier model from scratch 

## NB: EACH TRAIN YOU RUN, YOU HAVE FIRSTLY TO CHANGE THE ABOVE MENTIONED VARIABLES AND PATHS IN configuration.py (EXCEPT cfg.trained_folder)
## NB: If you change the batchsize, you must change the "nvid" variable in the loss_function.py file!

def save_model(model, json_path, weight_path, json=False):
    ''' 
    Save a trained model.
    - to_json(): returns a JSON string containing the network configuration
    - savemat(): saves a dictionary of names and arrays into a .mat file
    '''
    if json: 
        json_string = model.to_json()
        open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)

def load_model(json_path):
    '''
    Upload a trained model.
    - model_from_json(): parses a JSON model configuration string and returns a model instance'''
    model = model_from_json(open(json_path).read())
    return model

# Path to all features: we assume all videos' features are located in an unique folder  
path_all_features = cfg.path_all_features
# Give a name to your experiment 
train_name = cfg.train_exp_name

# Path where you want to save trained weights and model: subfolder in trained_models folder
output_dir = cfg.trained_folder+train_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
weights_path = os.path.join(output_dir, 'weights_'+train_name+'.mat')
model_path = os.path.join(output_dir, 'model_'+train_name+'.json')

# Initialize Fully Connected/LSTM Model 
adagrad = Adagrad(lr=0.01, epsilon=1e-08)
model = clf.classifier_model()
model.compile(loss=custom_objective, optimizer=adagrad)

# Understand if you've set all the variables correctly!
print('Model initialized: ')
if cfg.use_i3d == True:
    if cfg.use_lstm == True:
        print(' -------------> I3D - LSTM')
    else: 
        print(' -------------> I3D - Fully Connected')       
else:
    if cfg.use_lstm == True:
        print(' -------------> C3D - LSTM')
    else:
        print(' -------------> C3D - Fully Connected') 
        
# TRAINING 
print('Starting Training...') 
print('')
loss_graph = []
num_iters = 20000 
total_iterations = 0
batchsize = 60
time_before = datetime.now()

num_abn_vid = cfg.num_0
num_nor_vid = cfg.num_1

for it_num in range(num_iters):
    inputs, targets = load_dataset_Train_batch(path_all_features, path_all_features, batchsize, num_abn_vid, num_nor_vid) 
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    
    # Convert features extracted according to the necessary dimension to avoid errors from Tensorflow
    if cfg.use_lstm:
        if cfg.use_i3d:
            inputs = tf.reshape(inputs, [1920,1,1024])
        else: 
            inputs = tf.reshape(inputs, [1920,1,4096])
      
    targets = tf.convert_to_tensor(targets, dtype=tf.float32)
    batch_loss = model.train_on_batch(inputs, targets)
    loss_graph = np.hstack((loss_graph, batch_loss))
    total_iterations += 1
    
    if total_iterations % 100 == 0:
        # Print a message every 100th iteration 
        print('Iteration {} took: {}, with loss of {}'.format(str(total_iterations), str(datetime.now() - time_before), str(batch_loss)))
        print('')

    if total_iterations % 1000 == 0: 
        # Save model's weight + loss_graph every 1000th iteration
        weights_path2 = os.path.join(output_dir, 'weights_' + str(total_iterations) + '.mat')
        save_model(model, model_path, weights_path2)
        iteration_path = os.path.join(output_dir, 'iterations_graph_' + str(total_iterations) + '.mat')
        savemat(iteration_path, dict(loss_graph=loss_graph))

print('Successful Training - Model saved')
save_model(model, model_path, weights_path, json=True)
