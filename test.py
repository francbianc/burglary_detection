import os
import numpy as np
from classifier import *
from utils.visualization_util import *

# PATHS: cfg.C3D_path, cfg.I3D_path, cfg.score_path, cfg.classifier_model_weigts
# VARIABLES: cfg.use_i3d, cfg.use_lstm

def run_onlyTest():
    if cfg.use_i3d: 
        input_feat_path = cfg.I3D_path
    else: 
        input_feat_path = cfg.C3D_path
    
    for filename in os.listdir(input_feat_path):
        if filename.endswith('.txt'):
            video_features_name = os.path.join(input_feat_path, filename)
            name = os.path.basename(video_features_name).split('.')[0]
            print('Test on: {}'.format(name))

            # load bag features: array.shape = (32, 4096) or (32, 1024)
            rgb_feature_bag = np.loadtxt(video_features_name)
            dim = rgb_feature_bag.shape[1]
            if cfg.use_lstm:
                rgb_feature_bag = np.reshape(rgb_feature_bag, newshape=(32,1,dim))
                
            # initialize plain vanilla classifier 
            classifier_model = build_classifier_model()

            # classify using the pre-trained classifier model: len(predictions) = 32
            predictions = classifier_model.predict(rgb_feature_bag, batch_size=32)
            predictions = np.array(predictions).squeeze()
            print(predictions)

            save_path = os.path.join(cfg.score_path, name + '.txt')
            np.savetxt(save_path, predictions)

        
if __name__ == '__main__':
   run_onlyTest()