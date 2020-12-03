import scipy.io as sio
import configuration as cfg
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten
from tensorflow.keras.regularizers import L2

# PATHS: cfg.classifier_model_weigts 
# VARIABLES: cfg.use_lstm, cfg.use_i3d
# AIM: define the classifier model and upload its weights 

def classifier_model():
    """
    Create the classifier according to the specification in configuration.py
    Three possibilities:
    1. C3D - Fully Connected model: Features are extracted with C3D (32, 4096) and the classifier is not modified
    2. C3D - LSTM: Features are extracted with C3D (32, 4096) and the classifier has an additional LSTM layer (128,) 
    3. I3D - Fully Connected model: Features are extracted with I3D (32, 1024) and the classifier is not modified

    Returns
    ---------
    Sequential  
    """

    model = Sequential()
    if cfg.use_i3d == False: 
        if cfg.use_lstm:
            model.add(LSTM(128, input_shape=(1,4096)))
            model.add(Flatten())
            model.add(Dense(512, kernel_initializer='glorot_normal', kernel_regularizer=L2(0.001), activation='relu'))
        else:
            model.add(Dense(512, input_dim=4096, kernel_initializer='glorot_normal', kernel_regularizer=L2(0.001), activation='relu'))
    
    if cfg.use_i3d == True: 
        if cfg.use_lstm:
            model.add(LSTM(128, input_shape=(1,1024)))
            model.add(Flatten())
            model.add(Dense(512, kernel_initializer='glorot_normal', kernel_regularizer=L2(0.001), activation='relu'))
        else:
            model.add(Dense(512, input_dim=1024, kernel_initializer='glorot_normal', kernel_regularizer=L2(0.001), activation='relu'))
    
    model.add(Dropout(0.6))
    model.add(Dense(32, kernel_initializer='glorot_normal', kernel_regularizer=L2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(1, kernel_initializer='glorot_normal', kernel_regularizer=L2(0.001), activation='sigmoid'))
    return model

def build_classifier_model():
    """
    Create the model and load its weights, specified in configuration.py

    Returns
    ---------
    Sequential
        Model initialized with pre-trained weights
    """
    model = classifier_model()
    model = load_weights(model, cfg.classifier_model_weigts)
    return model

def conv_dict(dict2):
    """
    Retrieve the weights from a .mat file
    """
    diction = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                diction[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                diction[str(i)] = weights2
    return diction

def load_weights(model, weights_file):
    """
    Given a Keras model architecture and the corresponding weigths, create the pre-trained model that can be used to make predictions

    Parameters
    -------------
    model : Sequential
        Keras model architecture
    weights_file : .mat file
        .mat file in which weights are stored

    Returns
    ------------
    Sequential
        Pre-trained Keras model with corresponding weights
    """
    
    dict2 = sio.loadmat(weights_file)
    diction = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = diction[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

if __name__ == '__main__':
    model = build_classifier_model()
    model.summary()
