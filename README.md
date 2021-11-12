# PSPNet
This project contains my implementation of the PSPNet algorithm (Python, Tensorflow).

The data I used to train and test my model is a subset of the ADE20K dataset. It can be found at this address: https://www.kaggle.com/residentmario/ade20k-outdoors.

Some info about project files
Folders:
- utils folder contains useful function;
- the data folder must contain a zip file downloaded from kaggle;
- models is the folder for saved models;
- the layers folder contains several custom layers used in PSPNet;
- constants folder contains constants;
- logs folder contains training / evaluation information.

Scripts:
- data.py contains a class for data preparations and decoding model predictions (and some other functions for working with data);
- pspnet.py contains a class for creating the model;
- data_preparation.py is used to unzip data and save it to the appropriate folder;
- train_pspnet.py shows the code I used to train the model (train logs can be found in the logs folder);
- plot_predictions.py visualizes predictions;
- evaluate_performance.py evaluates model predictions (results can be found in the logs folder).

Results:
The most common metric for image segmentation is pixel-wise accuracy. My model achieves ~ 51% accuracy on a train set and ~ 53% accuracy on a test set. The modern model achieves 80% accuracy across the entire set. 
There are some reasons behind this difference:
- Authors of the paper use whole ADE20K dataset, while I use just a subset.
- I have limited resources. I trained my model using Colab GPU and even though I spent a lot of time optimizing my code to fit in memory limits. Also due to limited resources I used part of the ResNet50 instead of whole ResNet101. This results in bad model performance because most errors come from just 2 classes (3rd and 5th, look at train/test accuracy logs). I've tried to augment this classes and retrain the model but thats doesn't seem to work well, possibly it would be better to use bigger network.
