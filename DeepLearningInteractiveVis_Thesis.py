#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import data from Excel sheet
import pandas as pd
df = pd.read_excel('ADNI_komplett/ADNI combined.xlsx', sheet_name='sample')
#print(df)
sid = df['RID']
grp = df['Group at scan date (1=CN, 2=EMCI, 3=LMCI, 4=AD, 5=SMC)']
age = df['Age at scan']
sex = df['Sex (1=female)']
tiv = df['TIV']
field = df['MRI_Field_Strength']
grpbin = (grp > 1) # 1=CN, ...


# In[2]:


# Scan for nifti file names
import glob
dataAD = sorted(glob.glob('ADNI_komplett/AD/*.nii.gz'))
dataLMCI = sorted(glob.glob('ADNI_komplett/LMCI/*.nii.gz'))
dataCN = sorted(glob.glob('ADNI_komplett/CN/*.nii.gz'))
dataFiles = dataAD + dataLMCI + dataCN
numfiles = len(dataFiles)
print('Found ', str(numfiles), ' nifti files')


# In[3]:


# Match covariate information
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


# In[4]:


# Load residualized data from disk
import h5py
import numpy as np
from pandas import DataFrame
hf = h5py.File('ADNI_komplett/residuals.hdf5', 'r')
hf.keys # read keys
labels = np.array(hf.get('labels')) # note: was of data frame type before
images = np.array(hf.get('images'))
hf.close()
print(images.shape)


# In[5]:


# specify version of tensorflow
#%tensorflow_version 1.x
#%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)


# In[5]:


# Split data into training/validation and holdout test data
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
labels = to_categorical(np.asarray(labels))
# circumvent duplicate data
idx = np.asarray(range(numfiles))
tmp_idX,test_idX,tmp_Y,test_Y = train_test_split(idx, labels, test_size=0.1, stratify = labels, random_state=1)
train_idX,valid_idX,train_Y,valid_Y = train_test_split(tmp_idX, tmp_Y, test_size=0.1, stratify = tmp_Y, random_state=2)

testgrps = grps.iloc[test_idX, :]
print(testgrps) # prints diagnosis and RID
print('Distribution of diagnoses in holdout test data: [1=CN, 3=LMCI, 4=AD]')
print(testgrps.Group.value_counts())


# In[6]:


traindat = images[train_idX, :]
valdat = images[valid_idX, :]
testdat = images[test_idX, :]
train_label = train_Y


# In[7]:


testgrps["Group"] = testgrps["Group"].map({1:"CN", 3:"MCI", 4:"AD"})
Option_grps = np.array(testgrps)
Option_grps = Option_grps.astype('str')
Opt_grp = []

for i in range(len(Option_grps)):
    Opt_grp.append(' - ID '.join(Option_grps[i]))


#https://stackoverflow.com/questions/48279640/sort-a-python-list-while-maintaining-its-elements-indices  
Opt_grp = [(x, i) for (i, x) in enumerate(Opt_grp)]
Opt_grp  = sorted(Opt_grp)

def unzip(ls):
    if isinstance(ls, list):
        if not ls:
            return [], []
        else:
            Opt_grp, ys = zip(*ls)

        return list(Opt_grp), list(ys)  
    else:
        raise TypeError
sorted_xs, index_lst = unzip(Opt_grp)


# In[8]:


# Load CNN model from disk
from keras.models import load_model
mymodel = load_model('ADNI_komplett/model.hdf5')
#mymodel.summary()
#%whos


# In[10]:


#!pip install innvestigate


# In[9]:


# Load original images (background) from disk
import h5py
import numpy as np
hf = h5py.File('ADNI_komplett/orig_images.hdf5', 'r')
hf.keys # read keys
images_orig = np.array(hf.get('images'))
hf.close()
testdat_orig = images_orig[test_idX, :]
print(testdat_orig.shape)


# In[10]:


import innvestigate
import innvestigate.utils as iutils
import numpy as np
from matplotlib import pyplot as plt
from bokeh.models.widgets import Select
import numpy as np
import scipy
from bokeh.layouts import column, row
from bokeh.plotting import figure, curdoc
from bokeh.io import output_notebook, push_notebook, show
from bokeh.models.widgets import Slider
from bokeh.models.glyphs import Rect

model_wo_softmax = iutils.keras.graph.model_wo_softmax(mymodel)
#model_wo_softmax.summary()


# In[11]:


# see https://github.com/albermax/innvestigate/blob/master/examples/notebooks/imagenet_compare_methods.ipynb for a list of alternative methods
methods = [ # tuple with method,     params,                  label
#            ("deconvnet",            {},                      "Deconvnet"),
#            ("guided_backprop",      {},                      "Guided Backprop"),
#            ("deep_taylor.bounded",  {"low": -1, "high": 1},  "DeepTaylor"),
#            ("input_t_gradient",     {},                      "Input * Gradient"),
#            ("lrp.z",                {},                      "LRP-Z"),
#            ("lrp.epsilon",          {"epsilon": 1},          "LRP-epsilon"),
            ("lrp.alpha_1_beta_0",   {"neuron_selection_mode":"index"},  "LRP-alpha1beta0"),
]

# create analyzer -> only one selected here!
for method in methods:
  analyzer = innvestigate.create_analyzer(method[0], model_wo_softmax, **method[1])

# callback for a new subject being selected
def set_subject(subj_idx):
    global test_orig, test_img, pred, a # define global variables to store subject data
    test_img = testdat[subj_idx]
    test_img = np.reshape(test_img, (1,)+ test_img.shape) # add first subj index again to mimic original array structure
    test_orig = testdat_orig[subj_idx, :,:,:, 0]
    # evaluate/predict diag for selected subject
    pred = (mymodel.predict(test_img)[0,1]*100) # scale probability score to percent
    # derive relevance map from CNN model
    a = analyzer.analyze(test_img, neuron_selection=1)
    a = np.reshape(a, test_img.shape[1:4]) # drop first index again
    a = scipy.ndimage.filters.gaussian_filter(a, sigma=0.8) # smooth activity image
    # perform intensity normalization
    scale = np.quantile(np.absolute(a), 0.99)
    if scale==0: # fallback if quantile returns zero: directly use abs max instead
        scale = max(-np.amin(a), np.amax(a))
    a = (a/scale) # rescale range
    clipping_threshold = 3 # max value to be plotted, larger values will be set to this value; 
                            # corresponding to vmax in plt.imshow; vmin=-vmax used here
                            # value derived empirically here from the histogram of relevance maps
    a[a > clipping_threshold] = clipping_threshold # clipping of positive values
    a[a < -clipping_threshold] = -clipping_threshold # clipping of negative values
    a = a/clipping_threshold # final range: -1 to 1 float
    #print(np.max(a), np.min(a))
    # returns values by modifying global variables: test_orig, test_img, pred, a
    return
    
# Call once to initialize first image and variables
set_subject(index_lst[0]) # invoke with first subject


# In[12]:


subject_select = Select(title="Subjects:", value=sorted_xs[0], options=sorted_xs, width=200)
slice_slider = Slider(start=1, end=testdat.shape[3], value=10, step=1,
                  title="Coronal slice", width=200)
threshold_slider = Slider(start=0, end=1, value=0.4, step=0.05,
                  title="Relevance threshold", width=200)
clustersize_slider = Slider(start=0, end=200, value=10, step=10,
                  title="Minimum cluster size", width=200)
transparency_slider = Slider(start=0, end=1, value=0.3, step=0.05,title="Overlay Transparency:", width=200)
# initialize the figures
guide = figure(plot_width=208, plot_height=70, title='Relevance>threshold per slice:', toolbar_location=None,
              active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
guide.title.text_font = 'arial'
guide.title.text_font_style = 'normal'
#guide.title.text_font_size = '10pt'
guide.axis.visible = False
guide.x_range.range_padding = 0
guide.y_range.range_padding = 0
clusthist = figure(plot_width=208, plot_height=70, title='Distribution of cluster sizes:', toolbar_location=None,
              active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
clusthist.title.text_font = 'arial'
clusthist.title.text_font_style = 'normal'
clusthist.axis.visible = False
clusthist.x_range.range_padding = 0
clusthist.y_range.range_padding = 0
p = figure(plot_width=test_orig.shape[0]*6, plot_height=test_orig.shape[1]*5, title='',
          toolbar_location=None, output_backend="webgl",
          active_drag=None, active_inspect=None, active_scroll=None, active_tap=None)
p.axis.visible = False
p.x_range.range_padding = 0
p.y_range.range_padding = 0

# initialize column layout
layout = column(subject_select, guide, slice_slider, threshold_slider, clusthist, clustersize_slider,
                transparency_slider, p)

# for jupyter notebook:
#show(layout)
# alternatively, add layout to the document (for bokeh server)
curdoc().add_root(layout)
curdoc().title = 'Online AD brain viewer'


# In[13]:


from PIL import Image
from matplotlib import cm
from skimage.measure import label, regionprops

def apply_thresholds(map, threshold = 0.5, cluster_size = 20):
    global overlay, sum_pos, sum_neg, clust_sizes # define global variables to store subject data
    overlay = np.copy(map)
    overlay[np.abs(overlay) < threshold] = 0 # completely hide low values
    # cluster_size filtering
    labelimg = np.copy(overlay)
    labelimg[labelimg>0] = 1 # binarize img
    labelimg = label(labelimg, connectivity=2)
    lprops = regionprops(labelimg)
    clust_sizes = []
    for lab in lprops:
        clust_sizes.append(lab.area)
        if lab.area<cluster_size:
            labelimg[labelimg==lab.label] = 0 # remove small clusters
    labelimg[labelimg>0] = 1 # create binary mask
    np.multiply(overlay, labelimg, out=overlay)
    tmp = np.copy(overlay)
    tmp[tmp<0] = 0
    sum_pos = np.sum(tmp, axis=(0,1)) # sum of pos relevance in slices
    tmp = np.copy(overlay)
    tmp[tmp>0] = 0
    sum_neg = np.sum(tmp, axis=(0,1)) # sum of neg relevance in slices
    return overlay # result also stored in global variables: overlay, sum_pos, sum_neg

def bg2RGBA(bg):
    img = bg/4 + 0.25 # rescale to range of approx. 0..1 float
    img = np.uint8(cm.gray(img) * 255)
    return img

def overlay2RGBA(map, alpha = 0.5):
    # assume map to be in range of -1..1 with 0 for hidden content
    alpha_mask = np.copy(map)
    alpha_mask[np.abs(alpha_mask) > 0] = alpha # final transparency of visible content
    map = map/2 + 0.5 # range 0-1 float
    ovl = np.uint8(cm.jet(map) * 255) # cm translates range 0 - 255 uint to rgba array
    ovl[:,:,3] = np.uint8(alpha_mask * 255) # replace alpha channel (fourth dim) with calculated values
    return ovl


# In[18]:


# define other callback functions for the sliders

clust_hist_bins = list(range(0, 200+1, 10)) # list from (0, 10, .., 200); range max is slider_max_size+1
firstrun = True

def update_guide():
    curdoc().hold() # disable page updates
    global firstrun, pos_area, pos_line, neg_area, neg_line, hist
    x = np.arange(0, sum_neg.shape[0])
    y0 = np.zeros(x.shape, dtype=int)
    if firstrun:
        # plot everything
        guide.line(x, y0, color="#000000")
        pos_area = guide.varea(x=x, y1=sum_pos, y2=y0, fill_color ="#d22a40", fill_alpha =0.8, name="pos_area")
        pos_line = guide.line(x, y=sum_pos, line_width=2, color="#d22a40", name="pos_line")
        neg_area = guide.varea(x=x, y1=sum_neg, y2=y0, fill_color ="#36689b", fill_alpha =0.8, name="neg_area")
        neg_line = guide.line(x, y=sum_neg, line_width=2, color="#36689b", name="neg_line")
        # calc histogram; clip high values to slider max (=200)
        [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=clust_hist_bins)
        hist = clusthist.quad(bottom=np.zeros(histdat.shape, dtype=int), top=histdat, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="blue", name="hist")
        firstrun = False
    else:
        pos_line.data_source.data = {'x':x, 'y':sum_neg}
        pos_area.data_source.data = {'x':x, 'y1':sum_pos, 'y2':y0}
        neg_line.data_source.data = {'x':x, 'y':sum_neg}
        neg_area.data_source.data = {'x':x, 'y1':sum_neg, 'y2':y0}
        [histdat,edges] = np.histogram(np.clip(clust_sizes, a_min=None, a_max=200), bins=clust_hist_bins)
        hist.data_source.data = {'bottom':np.zeros(histdat.shape, dtype=int), 'top':histdat, 'left':edges[:-1], 'right':edges[1:]}
    curdoc().unhold() # enable page updates again

def plot():
    bg = test_orig[:,:,slice_slider.value-1]
    bg = bg2RGBA(bg)
    bg = np.flipud(bg)
    ovl = overlay[:,:,slice_slider.value-1]
    ovl = overlay2RGBA(ovl, alpha=1-transparency_slider.value)
    ovl = np.flipud(ovl)
    p.image_rgba(image=[bg,ovl], x=[0,0], y=[0,0], dw=[bg.shape[0],bg.shape[0]], dh=[bg.shape[1],bg.shape[1]])

def select_subject_callback(attr, old, new):
    set_subject( index_lst[sorted_xs.index(subject_select.value)] )
    p.title.text = "Scan predicted as %0.2f%% Alzheimer\'s" % pred
    apply_thresholds(a, threshold = threshold_slider.value, cluster_size = clustersize_slider.value)
    update_guide()
    plot()

def apply_thresholds_calback(attr, old, new):
    apply_thresholds(a, threshold = threshold_slider.value, cluster_size = clustersize_slider.value)
    update_guide()
    plot()

def set_slice_callback(attr, old, new):
    plot()

subject_select.on_change('value', select_subject_callback)
select_subject_callback('','','')

slice_slider.on_change('value', set_slice_callback)
threshold_slider.on_change('value', apply_thresholds_calback)
clustersize_slider.on_change('value', apply_thresholds_calback)
transparency_slider.on_change('value', set_slice_callback)


# In[19]:

subject_select.value=sorted_xs[0]



