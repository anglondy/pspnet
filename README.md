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
The most common metric for image segmentation is pixel-wise accuracy. My model achieves ~ 51% accuracy on a train set and ~ 53% accuracy on a test set. The state-of-the-art models achieve 80% accuracy across the entire set. 
There are several reasons for this difference:
- The authors of the article use the entire ADE20K dataset, and I only use a subset.
- I have limited resources. I trained my model with the Colab GPU and although I spent a lot of time optimizing my code to fit the memory constraints. Also due to limited resources, I used part of ResNet50 instead of all ResNet101. This leads to poor model performance because most errors come from just 2 classes (3rd and 5th, see training / test accuracy logs). I tried to augment these classes and retrain the model but this did not increase the performance of the network.
