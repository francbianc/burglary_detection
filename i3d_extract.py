import os
from i3d import *
from utils.visualization_util import *
import numpy as np

## USE THIS EXTENSION IF YOU'VE PUT THE VIDEOS YOU WANT TO USE INTO THE INPUT FOLDER

# PATHS: cfg.input_folder, cfg.I3D_path
# AIM: extract from 1 video, n features. Merge n features to end with 32 features and save them in a .txt file

def run_i3d():
    '''
    Starting from videos, extract features using the I3D model pre-trained on the Kinetics dataset and save them as .txt files.
    How?
        Each video is divided into n clips of 16 frames, where n = int(np.round(total_number_of_frames/16)).
        Each clip is given as input to the I3D model, that returns as output an array of 1024 floats. This array is 1 feature. 
        (n, 1024) features are then merged to end up with (32, 1024) features per each video. . 

    1. cfg.input_folder = folder containing all those videos whose features need to be extracted
    2. cfg.I3D_path = folder where these features are saved 
    '''
    for filename in os.listdir(cfg.input_folder):
        if filename.endswith('.mp4'):
            video_name = os.path.join(cfg.input_folder, filename)       
            name = os.path.basename(video_name).split('.')[0]          

            # Read video: create n clips of 16 frames
            video_clips, num_frames = get_video_clips(video_name)
            print("Number of clips in the video : ", len(video_clips))
            
            # Initialize I3D model 
            feature_extractor = i3d_feature_extractor()
            
            # Extract features
            rgb_features = []
            for i, clip in enumerate(video_clips):
                clip = np.array(clip)
                if len(clip) < params.frame_count:
                    continue

                clip = preprocess_input(clip)
                rgb_feature = feature_extractor.predict(clip)[0] 
                rgb_feature = rgb_feature.reshape(1024,)
                rgb_features.append(rgb_feature)
                print("Processed clip : {} / {}".format(i, len(video_clips)))
                
            # Bag features: from n to 32 features per each video
            rgb_features = np.array(rgb_features)
            rgb_feature_bag = interpolate(rgb_features, params.features_per_bag)
            
            # Save each bag of features as .txt 
            save_path = os.path.join(cfg.I3D_path, name + '.txt')
            np.savetxt(save_path, rgb_feature_bag)
        
        #print(filename)


if __name__ == '__main__':
   run_i3d()
