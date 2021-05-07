# Comparison-of-Convolutional-neural-network-training-parameters-for-detecting-Alzheimer-s-disease-and effect on visualization

Abstract:

Convolutional neural networks (CNN) have become a powerful tool for detecting patterns in image data. Recent
papers report promising results in the domain of disease detection using brain MRI data. Despite the high accuracy
obtained from CNN models for MRI data so far, almost no papers provided information on the features
or image regions driving this accuracy as adequate methods were missing or challenging to apply. Recently, the
toolbox iNNvestigate has become available, implementing various state of the art methods for deep learning visualizations.
Currently, there is a great demand for a comparison of visualization algorithms to provide an overview
of the practical usefulness and capability of these algorithms.
Therefore, this thesis has two goals:
1. To systematically evaluate the influence of CNN hyper-parameters on model accuracy
2. To compare various visualization methods with respect to the quality (i.e. randomness/focus, soundness)

Visualization of relevant areas in the MRI scan relevant to the classification of Alzheimer's disease.

Bokeh server is run on local machine for the visualization purpose.
Here we can select the subject and the coronal slice for which we can visualize the relevant areas that the model considers for the classification.
The hippocampus region highlighted by the model corresponds to the major region attrophied in the case of Alzheimer's disease in medical literature.

![](MRI-relevance-map-visualization-using-Bokeh.gif)
