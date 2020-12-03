import os
import numpy as np
from classifier import *
from utils.visualization_util import *
import configuration as cfg

## USE THIS EXTENSION IF YOU'VE USED ONLY FEW VIDEOS SO FAR

# PATHS: cfg.path_all_features, cfg.NamesAnn_path, cfg.classifier_model_weigts
# VARIABLES: cfg.train_exp_name, cfg.use_i3d, cfg.use_lstm
# AIM: given features extracted and weights saved from a training experiment, compute the predicted labels for test data 

path_all_features = cfg.path_all_features
experiment_name = cfg.train_exp_name 
# The experiment name shuould be in line with the name of the training experiment, whose weights will be used to compute the predictions

def run_onlyTest():
    """
    Predict if a video segment is normal or abnormal by calculating the anomaly score for each of its 32 features.
    Predictions are made using a pre-trained networks built and trained according to the specifications in configuration.py.
    
    Returns
    -----------
    Save predictions of each video as .txt files in a dedicated folder. 
    Predictions are arrays of 32 floats.
    """

    # Features have been extracted either with I3D or C3D network
    feat_output_path = [line.strip().replace('.mp4', '.txt').replace('/', '_Features/') for line in open(cfg.NamesAnn_path, 'r')]
    if cfg.use_i3d:
        feat_output_path = [line.strip().replace('.mp4', '.txt').replace('/', '_Features_I3D/') for line in open(cfg.NamesAnn_path, 'r')]
    
    feat_output_path = [os.path.join(path_all_features, i) for i in feat_output_path]
    
    # We have 310 videos in the test set
    assert len(feat_output_path) == 310

    # Create a folder to store the predictions 
    score_output_path = os.path.join(path_all_features, 'Scores_'+experiment_name)
    if not os.path.exists(score_output_path):
        os.makedirs(score_output_path)

    # Loop over all the videos in the test set
    for filename in feat_output_path:
        if filename.endswith('.txt'):
            video_features_name = filename
            name = os.path.basename(video_features_name).split('.')[0]
            print('Test on: {}'.format(name))

            # Load bag features: array.shape = (32, 4096) or (32, 1024) 
            rgb_feature_bag = np.loadtxt(video_features_name)
            dim = rgb_feature_bag.shape[1]
            if cfg.use_lstm:
                rgb_feature_bag = np.reshape(rgb_feature_bag, newshape=(32,1,dim))
        
            # Initialize classifier 
            classifier_model = build_classifier_model()

            # Classify using the pre-trained classifier model: len(predictions) = 32
            predictions = classifier_model.predict(rgb_feature_bag, batch_size=32)
            predictions = np.array(predictions).squeeze()
            #print(predictions)

            # Save the predictions for each video for future use
            save_path = os.path.join(score_output_path, name + '.txt')
            np.savetxt(save_path, predictions)

        
if __name__ == '__main__':
   run_onlyTest()
