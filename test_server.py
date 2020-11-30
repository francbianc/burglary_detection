import os
import numpy as np
from classifier import *
from utils.visualization_util import *
import configuration as cfg

# PATHS: cfg.path_all_features, cfg.NamesAnn_path, cfg.classifier_model_weigts
# VARIABLES: cfg.train_exp_name, cfg.use_i3d, cfg.use_lstm

path_all_features = cfg.path_all_features
experiment_name = cfg.train_exp_name

def run_onlyTest():
    """
    Predict if a video segment is normal or anomalous by calculating the anomaly score for each of the 32 temporal segments of the video bag.
    Predictions are made using a pre-trained networks built and trained according to the specifications in configuration.py

    This script should be run on a server.

    Returns
    -----------
    Save predictions of each video as .txt files in a dedicated folder
    """

    # features can be extracted either with I3D or C3D network
    feat_output_path = [line.strip().replace('.mp4', '.txt').replace('/', '_Features/') for line in open(cfg.NamesAnn_path, 'r')]
    if cfg.use_i3d:
        feat_output_path = [line.strip().replace('.mp4', '.txt').replace('/', '_Features_I3D/') for line in open(cfg.NamesAnn_path, 'r')]
    
    feat_output_path = [os.path.join(path_all_features, i) for i in feat_output_path]
    
    # we have 310 videos in the test set
    assert len(feat_output_path) == 310

    # create folder to store the prediction if does not exist
    score_output_path = os.path.join(path_all_features, 'Scores_'+experiment_name)
    if not os.path.exists(score_output_path):
        os.makedirs(score_output_path)

    # loop over all the videos in the test set
    for filename in feat_output_path:
        if filename.endswith('.txt'):
            video_features_name = filename
            name = os.path.basename(video_features_name).split('.')[0]
            print('Test on: {}'.format(name))

            # load bag features: array.shape = (32, 4096) or (32, 1024) 
            rgb_feature_bag = np.loadtxt(video_features_name)
            dim = rgb_feature_bag.shape[1]
            if cfg.use_lstm:
                rgb_feature_bag = np.reshape(rgb_feature_bag, newshape=(32,1,dim))
        
            # initialize classifier 
            classifier_model = build_classifier_model()

            # classify using the pre-trained classifier model: len(predictions) = 32
            predictions = classifier_model.predict(rgb_feature_bag, batch_size=32)
            predictions = np.array(predictions).squeeze()
            #print(predictions)

            # save the predictions for each video for future use
            save_path = os.path.join(score_output_path, name + '.txt')
            np.savetxt(save_path, predictions)

        
if __name__ == '__main__':
   run_onlyTest()