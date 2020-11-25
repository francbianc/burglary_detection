# ----- NOT MODIFY THESE PATHS 
input_folder = './Input'
C3D_path = './C3D_Features'
C3D_info_path = './C3D_Features_Info'
I3D_path = './I3D_Features'

trained_folder = './trained_models/train_'
score_path = './Scores'
all_ann_path = './training_splits/Sultani/test_annot310_sultani.txt'

# ----- MODIFY THESE VARIABLES
use_lstm = True
use_i3d = True

## Weights of the pre-trained classifier (10 possibilities)
classifier_model_json = './trained_models/model.json'
#classifier_model_weigts = './trained_models/weights_L1L2.mat'
#classifier_model_weigts = './trained_models/train_exp1_C3D/weights_exp1_C3D.mat'
#classifier_model_weigts = './trained_models/train_exp2_C3D/weights_exp2_C3D.mat'
#classifier_model_weigts = './trained_models/train_exp1_128LSTM_C3D/weights_exp1_128LSTM_C3D.mat'
classifier_model_weigts = './trained_models/train_exp2_128LSTM_C3D/weights_exp2_128LSTM_C3D.mat'
#classifier_model_weigts = './trained_models/train_exp1_I3D/weights_exp1_I3D.mat'
#classifier_model_weigts = './trained_models/train_exp2_I3D/weights_exp2_I3D.mat'
#classifier_model_weigts = './trained_models/train_sul_C3D/weights_sul_C3D.mat'
#classifier_model_weigts = './trained_models/train_sul_128LSTM_C3D/weights_sul_128LSTM_C3D.mat'
#classifier_model_weigts = './trained_models/train_sul_I3D/weights_sul_I3D.mat'

## Paths to use the entire UCF_Crimes dataset 
path_all_videos = '/home/3022790/UCF_Crimes/Videos/'        
path_all_features = '/home/3022790/UCF_Crimes/Videos/' 

## Name any experiment you want to perform with train.py
train_exp_name = 'Experiment ...'      

## Choose which training-test set you prefer (3 possibilities)
## Sultani (without big videos)
#train_0_path = './training_splits/Sultani/train_names810_sultani_class0.txt'
#train_1_path = './training_splits/Sultani/train_names800_sultani_class1.txt'
#num_0 = 810
#num_1 = 800
#Ann_path = './training_splits/Sultani/test_annot310_sultani.txt'
#NamesAnn_path = './training_splits/Sultani/test_names310_sultani.txt'

## Experiment 1
#train_0_path = './training_splits/Experiment1/train_names750_exp1_class0.txt'
#train_1_path = './training_splits/Experiment1/train_names750_exp1_class1_nobig.txt'
#num_0 = 750
#num_1 = 750
#Ann_path = './training_splits/Experiment1/test_annot310_exp1.txt'
#NamesAnn_path = './training_splits/Experiment1/test_names310_exp1.txt'

## Experiment 2
train_0_path = './training_splits/Experiment2/train_names750_exp2_class0.txt'
train_1_path = './training_splits/Experiment2/train_names750_exp2_class1_nobig.txt'
num_0 = 750
num_1 = 750
Ann_path = './training_splits/Experiment2/test_annot310_exp2.txt'
NamesAnn_path = './training_splits/Experiment2/test_names310_exp2.txt'