import numpy as np
import cv2
import configuration as cfg

import tensorflow as tf 
#print(tf.version.VERSION)
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv3D, MaxPool3D, ZeroPadding3D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.utils import get_file

# PATHS: C3D_MEAN_PATH, C3D_SPORTS1M_PATH
# AIM: define the C3D model and upload the weights trained on the Sports-1M dataset, that consists of 1.1 million sports videos.

C3D_MEAN_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/c3d_mean.npy'
C3D_SPORTS1M_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5'

def preprocess_input(video):
    intervals = np.ceil(np.linspace(0, video.shape[0] - 1, 16)).astype(int)
    frames = video[intervals]

    # Reshape to 128x171
    reshape_frames = np.zeros((frames.shape[0], 128, 171, frames.shape[3]))
    for i, img in enumerate(frames):
        img = cv2.resize(img, dsize=(171, 128), interpolation=cv2.INTER_CUBIC)
        reshape_frames[i, :, :, :] = img

    mean_path = get_file('c3d_mean.npy',
                         C3D_MEAN_PATH,
                         cache_subdir='models',
                         md5_hash='08a07d9761e76097985124d9e8b2fe34')

    mean = np.load(mean_path)
    # Normalize frames
    reshape_frames -= mean
    # Crop to 112x112
    reshape_frames = reshape_frames[:, 8:120, 30:142, :]
    # Add extra dimension for samples
    reshape_frames = np.expand_dims(reshape_frames, axis=0)

    return reshape_frames


def C3D(weights='sports1M'):
    if weights not in {'sports1M', None}:
        raise ValueError('weights should be either be sports1M or None')

    if K.image_data_format() == 'channels_last':
        shape = (16, 112, 112, 3)
    else:
        shape = (3, 16, 112, 112)

    model = Sequential()
    model.add(Conv3D(64, 3, activation='relu', padding='same', name='conv1', input_shape=shape))
    model.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same', name='pool1'))

    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv2'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2'))

    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3b'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3'))

    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4b'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4'))

    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5'))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    if weights == 'sports1M':
        # Load weights
        c3d_model_weights = get_file('c3d_sports1m.h5', C3D_SPORTS1M_PATH)
        model.load_weights(c3d_model_weights)

    return model


def c3d_feature_extractor():
    model = C3D()
    # Extract visual features from the fully connected (FC) layer FC6 of the C3D network
    layer_name = 'fc6'
    feature_extractor_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return feature_extractor_model
