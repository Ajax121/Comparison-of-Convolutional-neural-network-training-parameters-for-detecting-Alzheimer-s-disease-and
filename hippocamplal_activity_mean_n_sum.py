# -*- coding: utf-8 -*-
"""hippocamplal activity mean n sum.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15PCagTk3zmKZcEteqLbEe20kcq2EcwGZ
"""

# Loading the Google drive/folder into Colab
from google.colab import drive
drive.mount('/content/drive')
!ls -lh /content/drive/My\ Drive/ADNI_komplett

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

# Import data from Excel sheet
import pandas as pd
df = pd.read_excel('/content/drive/My Drive/ADNI_komplett/ADNI combined.xlsx', sheet_name='sample')
#print(df)
sid = df['RID']
grp = df['Group at scan date (1=CN, 2=EMCI, 3=LMCI, 4=AD, 5=SMC)']
age = df['Age at scan']
sex = df['Sex (1=female)']
tiv = df['TIV']
field = df['MRI_Field_Strength']
grpbin = (grp > 1) # 1=CN, ...

# Scan for nifti file names
import glob
dataAD = sorted(glob.glob('/content/drive/My Drive/ADNI_komplett/AD/*.nii.gz'))
dataLMCI = sorted(glob.glob('/content/drive/My Drive/ADNI_komplett/LMCI/*.nii.gz'))
dataCN = sorted(glob.glob('/content/drive/My Drive/ADNI_komplett/CN/*.nii.gz'))
dataFiles = dataAD + dataLMCI + dataCN
numfiles = len(dataFiles)
print('Found ', str(numfiles), ' nifti files')

import re
debug = False
cov_idx = [-1] * numfiles # list; array: np.full((numfiles, 1), -1, dtype=int)
print('Matching covariates for loaded files ...')
for i,id in enumerate(sid):
    p = [j for j,x in enumerate(dataFiles) if re.search('_%04d_' % id, x)] # translate ID numbers to four-digit numbers, get both index and filename
    if len(p)==0:
        if debug: print('Did not find %04d' % id) # did not find Excel sheet subject ID in loaded file selection
    else:
        if debug: print('Found %04d in %s: %s' % (id, p[0], dataFiles[p[0]]))
        cov_idx[p[0]] = i # store Excel index i for data file index p[0]
print('Checking for scans not found in Excel sheet: ', sum(x<0 for x in cov_idx))

labels = pd.DataFrame({'Group':grpbin}).iloc[cov_idx, :]
grps = pd.DataFrame({'Group':grp, 'RID':sid}).iloc[cov_idx, :]

#Load residualized data from disk
import h5py
import numpy as np
hf = h5py.File('/content/drive/My Drive/wb_orig+label.hdf5', 'r')
hf.keys # read keys
labels = np.array(hf.get('labels')) # note: was of data frame type before
images = np.array(hf.get('images'))
hf.close()
images.shape

# Display a single scan
from matplotlib import pyplot as plt
#import numpy as np
test_img = images[0, :,:,:, 0]
ma = np.max(test_img)
mi = np.min(test_img)
test_img = (test_img - mi) / (ma - mi) # normalising to (0-1) and then normalising to 0 mean and 1 std
#test_img = (test_img - test_img.mean())/test_img.std() # normalizing by mean and sd
print('displaying image ', dataFiles[0])
for i in range(test_img.shape[2]):
  if (i % 10 == 0): # only display each tenth slice
    plt.figure()
    a = test_img[:,:,i]
    plt.imshow(a, cmap='gray')

import tensorflow as tf
print(tf.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

!unzip "/content/drive/My Drive/models.zip"

# Load CNN model from disk
from keras.models import load_model
#mymodel = load_model('/content/drive/My Drive/ADNI_komplett/model_wb.hdf5')
#mymodel = load_model('/content/32-slice_Models_20-filters/32slice_model_5CNN_layers_20_1FC_fold4.hdf5')
mymodel = load_model('/content/wb_3cnn_no_residual_2xDA.hdf5',compile=False)
mymodel.summary()

grps.RID.to_numpy(dtype=np.int)

#Load original images (background) from disk
import h5py
import numpy as np
hf = h5py.File('/content/drive/My Drive/ADNI_komplett/orig_images.hdf5', 'r')
hf.keys # read keys
images_orig = np.array(hf.get('images'))
hf.close()
#testdat_orig = images_orig[test_idX, :]

!mkdir Hippo_overlay_latest

!unzip "/content/drive/My Drive/Hippo_overlay_latest.zip" -d "/content/Hippo_overlay_latest"

import nibabel as nib

# define FOV to reduce required memory size
x_range_from = 13; x_range_to = 107
y_range_from = 15; y_range_to = 126 # full brain: 15:126  32 slice 54:86
z_range_from = 12; z_range_to = 100

data_overlay = sorted(glob.glob('Hippo_overlay_latest/*.nii.gz'))
hippo_overlay = np.zeros((len(data_overlay), z_range_to-z_range_from, x_range_to-x_range_from, y_range_to-y_range_from, 1), dtype=np.float32) # numfiles× z × x × y ×1; avoid 64bit types

for i in range(len(data_overlay)):   
    img = nib.load(data_overlay[i])
    img = img.get_data()[x_range_from:x_range_to, y_range_from:y_range_to, z_range_from:z_range_to]
    img = np.transpose(img, (2, 0, 1)) # reorder dimensions to match coronal view z*x*y in MRIcron etc.
    img = np.flip(img) # flip all positions
    hippo_overlay[i, :,:,:, 0] = np.nan_to_num(img)

###### check which file corresponds to hippo l,r and both with indexes 0,1 and 2#####
data_overlay[2]

#hippos
hippo_both = np.zeros((z_range_to-z_range_from, x_range_to-x_range_from, y_range_to-y_range_from), dtype=np.float32)
hippo_left = np.zeros((z_range_to-z_range_from, x_range_to-x_range_from, y_range_to-y_range_from), dtype=np.float32)
hippo_right = np.zeros((z_range_to-z_range_from, x_range_to-x_range_from, y_range_to-y_range_from), dtype=np.float32)
hippo_both[:,:,:] = hippo_overlay[0,:,:,:,0]
hippo_left[:,:,:] = hippo_overlay[1,:,:,:,0]
hippo_right[:,:,:] = hippo_overlay[2,:,:,:,0]

!pip install innvestigate

#!pip install innvestigate
import innvestigate
import innvestigate.utils as iutils
import numpy as np
from matplotlib import pyplot as plt
import scipy

model_wo_softmax = iutils.keras.graph.model_wo_softmax(mymodel)
#model_wo_softmax.summary()


# see https://github.com/albermax/innvestigate/blob/master/examples/notebooks/imagenet_compare_methods.ipynb for a list of alternative methods
methods = [ # tuple with method,     params,                  label
#            ("deconvnet",            {},                      "Deconvnet"),
#            ("guided_backprop",      {},                      "Guided Backprop"),
#            ("deep_taylor.bounded",  {"low": -1, "high": 1},  "DeepTaylor"),
#            ("input_t_gradient",     {},                      "Input * Gradient"),
#            ("lrp.z",                {},                      "LRP-Z"),
#            ("lrp.epsilon",          {"epsilon": 1},          "LRP-epsilon"),
            ("lrp.alpha_1_beta_0",   {"neuron_selection_mode":"index"},                      "LRP-alpha1beta0"),
]

# create analyzer
analyzers = []
for method in methods:
    #analyzer = innvestigate.create_analyzer("deep_taylor.bounded", model_wo_softmax, **params )
    analyzer = innvestigate.create_analyzer(method[0], model_wo_softmax, **method[1])
    # Some analyzers require training.
  #   analyzer.fit(test_img, batch_size=30, verbose=1)
  #  analyzers.append(analyzer)

print("subject_ID Sum_activation_of_right_hippocampal_volume Sum_activation_of_left_hippocampal_volume Sum_activation_of_both_hippocampal_volume ")
#subj_idx = 9 # good visualizations for subjects idx 4 (AD), 5 (AD), 6 (LMCI), 8 (AD), 10 (LMCI), 27 (CN)
for indx in range(len(grps)):
    test_img = images[indx]
    #test_orig = images_orig[indx]
    #print('test image for subject of binary group: %d' % test_Y[subj_idx, 1]) # first col will indicate CN, second col indicates MCI/AD
    #print('test image for subject of ADNI diagnosis: %d [1-CN, 3-LMCI, 4-AD]' % testgrps.Group.to_numpy(dtype=np.int)[subj_idx])
    
    ####print('test subject ID %s' % grps.RID.to_numpy(dtype=np.int)[indx])

    test_img = np.reshape(test_img, (1,)+ test_img.shape) # add first subj index again to mimic original array structure
    #test_orig = np.reshape(test_orig, (1,)+ test_orig.shape) # add first subj index again to mimic original array structure

    #for method,analyzer in zip(methods, analyzers):
    a = np.reshape(analyzer.analyze(test_img,neuron_selection=1), test_img.shape[1:4])
    #"""
    np.clip(a,a_min=0,a_max=None, out=a)
    a = scipy.ndimage.filters.gaussian_filter(a, sigma=0.8) # smooth activity image
    scale = np.quantile(np.absolute(a), 0.99)
    if scale==0:
        scale = max(np.amax(a))   #scale = max(-np.amin(a), np.amax(a))
            #print(scale)
    a = (a/scale)
    #"""
    #a = (a - np.min(a)) / (np.max(a) - np.min(a)) 
    overlay_act_both = hippo_both * a
    overlay_act_l = hippo_left * a
    overlay_act_r = hippo_right * a
        
    print(grps.RID.to_numpy(dtype=np.int)[indx],np.sum(overlay_act_r),np.sum(overlay_act_l),np.sum(overlay_act_both))

        #print(grps.RID.to_numpy(dtype=np.int)[indx],np.mean(overlay_act_r[hippo_right>0]),np.sum(overlay_act_r[hippo_right>0]),np.mean(overlay_act_l[hippo_left>0]),np.sum(overlay_act_l[hippo_left>0]),np.mean(overlay_act_both[hippo_both>0]),np.sum(overlay_act_both[hippo_both>0]))
        #print('subject ID %s : Mean activation of left hippocampal volume %f : Sum activation of left hippocampal volume %f' % (grps.RID.to_numpy(dtype=np.int)[indx],np.mean(overlay_act_l[hippo_left>0]),np.sum(overlay_act_l[hippo_left>0])))
        #print('subject ID %s : Mean activation of both hippocampal volume %f : Sum activation of both hippocampal volume %f' % (grps.RID.to_numpy(dtype=np.int)[indx],np.mean(overlay_act_both[hippo_both>0]),np.sum(overlay_act_both[hippo_both>0])))

