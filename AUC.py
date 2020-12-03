import numpy as np
import os
from utils.visualization_util import *
from sklearn import metrics
import configuration as cfg

## USE THIS EXTENSION IF YOU'VE PUT THE VIDEOS YOU WANT TO USE INTO THE INPUT FOLDER 

# PATHS: cfg.all_ann_path, cfg.input_folder, cfg.score_path, cfg.C3D_path, cfg.I3D_path
# VARIABLES: cfg.use_i3d
# AIM: compute AUC for test data 

no_video = 1
frm_counter = 0
All_Detect = np.zeros(1000000)
All_GT = np.zeros(1000000)

# Get the true scores for each test video
all_annotations = [line.strip().split() for line in open(cfg.all_ann_path, 'r')]
all_annotations_dict = {l[0].split('.')[0]:l[-4:] for l in all_annotations}

# Get the number of frames for each test video
all_frames = [i.strip().split() for i in open('./num_frames.txt', 'r')]
all_frames_dict = {l[0].split('/')[1]:l[1:] for l in all_frames}

for filename in os.listdir(cfg.input_folder):
   if filename.endswith('.mp4'):
      video_name = os.path.join(cfg.input_folder, filename)
      name = os.path.basename(video_name).split('.')[0]
      print(name)
      name_txt = name + '.txt'
      
      scores = os.path.join(cfg.score_path, name_txt)
      score = [line.strip() for line in open(scores, 'r')]
      # list of 32 str (each str contains 1 score) 
      
      if cfg.use_i3d:
         I3D_files = os.path.join(cfg.I3D_path, name_txt)
         I3D_file = [line.strip() for line in open(I3D_files, 'r')]
         # list of 32 str (each str contains 1024 floats)
      else: 
         C3D_files = os.path.join(cfg.C3D_path, name_txt)
         C3D_file = [line.strip() for line in open(C3D_files, 'r')]
         # list of 32 str (each str contains 4096 floats)

      Ann = all_annotations_dict[name] 
      # list of 4 str (each str contains 1 annot as str)

      num_frames = int(all_frames_dict[name_txt][0])
      # integer
      
      # Assign to each frame the anomaly score of the feature it belongs to
      num_features = int(np.round(num_frames/16))
      num_frames_C3D = num_features*16 # As the features were computed for every 16 frames
      Detection_score_32shots = np.zeros(num_frames_C3D)
      Thirty2_shots = np.round(np.linspace(0, num_features, 32))

      l = range(len(Thirty2_shots))
      p_c = -1
      for c_shots, n_shots in zip (l, l[1:]):
         p_c=p_c+1
         ss=Thirty2_shots[c_shots]
         ee=Thirty2_shots[n_shots]-1
         #print('ss:', ss, 'ee:', ee)
         #print('c_shots:', c_shots, 'n_shots:', n_shots)

         if c_shots==len(Thirty2_shots):
            ee=Thirty2_shots[n_shots]

         if ee<ss:
            Detection_score_32shots[(int(ss))*16:(int(ss))*16+16+1]=score[p_c]
            #print(ee<ss)
         else:
            Detection_score_32shots[(int(ss))*16:(int(ee))*16+16+1]=score[p_c]
            #print(ee>ss)

      #print(num_frames)
      #print(len(Detection_score_32shots))
      # Assign to the last frames of a video the 32th score 
      if num_frames > len(Detection_score_32shots):
         Final_score = np.append(Detection_score_32shots, np.repeat(Detection_score_32shots[-1], [num_frames-len(Detection_score_32shots)]))
         GT=np.zeros(num_frames)
      else:
         Final_score = Detection_score_32shots
         GT=np.zeros(len(Detection_score_32shots))

      # Check the temporal annotation
      t_txt = [int(i) for i in Ann]
      
      for y in range(0,3,2):
         if t_txt[y] >= 0:
            st_fr = max(int(float(t_txt[y])), 0)
            end_fr = min(int(float(t_txt[y+1])), num_frames)
            GT[st_fr:end_fr+1] = 1

      All_Detect[frm_counter:frm_counter+len(Final_score)]=Final_score
      All_GT[frm_counter:frm_counter+len(Final_score)]=GT
      print('Video ', no_video, ' successfully processed!')
      no_video = no_video + 1
      frm_counter=frm_counter+len(Final_score)

All_Detect = All_Detect[0:frm_counter]
All_GT = All_GT[0:frm_counter]

# Compute the AUC with the Scikit-Learn function
fpr, tpr, thresholds = metrics.roc_curve(All_GT, All_Detect)
AUC = metrics.auc(fpr, tpr)
print('AUC: ', AUC)
