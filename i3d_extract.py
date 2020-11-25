import os
from i3d import *
from utils.visualization_util import *
import numpy as np

# PATHS: cfg.input_folder, cfg.I3D_path

def run_i3d():
    '''
    1. cfg.input_folder = folder containing all those videos whose features need to be extracted
    2. cfg.I3D_path = folder where these features are saved 
    '''
    for filename in os.listdir(cfg.input_folder):
        if filename.endswith('.mp4'):
            video_name = os.path.join(cfg.input_folder, filename)       
            name = os.path.basename(video_name).split('.')[0]          

            # read video
            video_clips, num_frames = get_video_clips(video_name)
            print("Number of clips in the video : ", len(video_clips))
            
            # initialize I3D model 
            feature_extractor = i3d_feature_extractor()
            
            # extract features
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
                
            # bag features
            rgb_features = np.array(rgb_features)
            rgb_feature_bag = interpolate(rgb_features, params.features_per_bag)
            
            save_path = os.path.join(cfg.I3D_path, name + '.txt')
            np.savetxt(save_path, rgb_feature_bag)
        
        #print(filename)


if __name__ == '__main__':
   run_i3d()