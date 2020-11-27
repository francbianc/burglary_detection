import os
from os import listdir
from datetime import datetime
import numpy as np
import configuration as cfg

# PATHS: cfg.train_0_path, cfg.train_1_path
# VARIABLES: server, cfg.use_i3d

def load_dataset_Train_batch(Abnormal_Features_Path, Normal_Features_Path, batch_size, num_abnorm, num_norm, server=True, use_i3d=cfg.use_i3d):
    '''AbnormalPath, NormalPath: paths of the folder containing extracted features (such as C3D_Features, I3D_Features or 
    any other folder containing all the features divided by category in subfolders)
    '''
    batchsize = batch_size
    n_exp = int(batchsize / 2)

    # Number of label_1 (normal) and label_0 (abnormal) videos in training set 
    Num_abnormal = num_abnorm  
    Num_normal = num_norm    

    # Randomly select 60 videos to put in the batch
    Abnor_list_iter = np.random.permutation(Num_abnormal)
    Abnor_list_iter = Abnor_list_iter[Num_abnormal - n_exp:]
    Norm_list_iter = np.random.permutation(Num_normal)
    Norm_list_iter = Norm_list_iter[Num_normal - n_exp:]

    print("Loading Abnormal videos' features...")
    All_Videos0 = [i.strip() for i in open(cfg.train_0_path, 'r')]
    if server: 
        if use_i3d:
            All_Videos0 = [i.strip().replace('_Features/', '_Features_I3D/') for i in open(cfg.train_0_path, 'r')]
    assert len(All_Videos0) == Num_abnormal
    AllFeatures = []
    Video_count = -1

    if use_i3d:
        dim = 1024
    else: 
        dim = 4096 

    for iv in Abnor_list_iter:
        Video_count = Video_count + 1
        VideoPath = os.path.join(Abnormal_Features_Path, All_Videos0[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        num_feat = len(words) / dim
        count = -1
        VideoFeatues = []
        for feat in range(0, int(num_feat)):
            feat_row1 = np.float32(words[feat * dim:feat * dim + dim])
            count = count + 1
            if count == 0:
                VideoFeatues = feat_row1
            if count > 0:
                VideoFeatues = np.vstack((VideoFeatues, feat_row1))
        
        if Video_count == 0:
            AllFeatures = VideoFeatues
        if Video_count > 0:
            AllFeatures = np.vstack((AllFeatures, VideoFeatues))
            # AllFeatures: array.shape = (32*30, 4096) or (32*30, 1024)
    print(" >> Abnormal Features loaded")

    print("Loading Normal videos' features...")
    All_Videos1 = [i.strip() for i in open(cfg.train_1_path, 'r')]
    if server: 
        if use_i3d:
            All_Videos1 = [i.strip().replace('_Features/', '_Features_I3D/') for i in open(cfg.train_1_path, 'r')]
    assert len(All_Videos1) == Num_normal

    for iv in Norm_list_iter:
        VideoPath = os.path.join(Normal_Features_Path, All_Videos1[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        feat_row1 = np.array([])
        num_feat = len(words) / dim
        count = -1
        VideoFeatues = []
        for feat in range(0, int(num_feat)):
            feat_row1 = np.float32(words[feat * dim:feat * dim + dim])
            count = count + 1
            if count == 0:
                VideoFeatues = feat_row1
            if count > 0:
                VideoFeatues = np.vstack((VideoFeatues, feat_row1))
            feat_row1 = []
        AllFeatures = np.vstack((AllFeatures, VideoFeatues))
        # AllFeatures: array.shape = (32*60, 4096) or (32*60, 1024)
    print(" >> Normal Features loaded")

    AllLabels = np.zeros(32 * batchsize, dtype='uint8')
    th_loop1 = n_exp * 32       # first 960 (32*30) ABNORM features
    th_loop2 = n_exp * 32 - 1   # last 960 (32*30) NORM features

    for iv in range(0, 32 * batchsize):
        if iv < th_loop1:               # 0-959 ABNORMAL
            AllLabels[iv] = float(0)    # label = 0
        if iv > th_loop2:               # 960-1920 NORMAL
            AllLabels[iv] = float(1)    # label = 1
    # AllLabels: array of 1920 floats

    print(" >> Abnormal + Normal Labels loaded")
    return AllFeatures, AllLabels