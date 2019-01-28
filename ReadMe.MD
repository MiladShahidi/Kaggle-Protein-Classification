# Image Classification with Deep Convolutional Neural Networks

![caption]("images/proteins.png")

In this project, I use data from Kaggle's Human Protein Classification and train a deep CNN to identify various types of protein in microscopic images. The model achieves an F1 score of 0.471 on the test set, which would be ranked in the top 15% of the participants.

There are 28 different of types of proteins in the dataset and each image may contain more than one type of protein. Since this is a multi-class and multi-label problem, I slightly modify the standard ResNet50 architecture, to suit the multi-label nature of the problem. More specifically, I use a binary cross enthropy loss function and traet the problem as 28 separate binary classification problems.

I use random mirroring and rotation to augment the training data. My experiments indicate that this improves the performance of the model significantly. To address the problem of imbalance in the data, I optimaize the classification threshold for each class by searching for the empirically optimal threshold in the validation data.

The complete Tensorflow training package is available [here](https://github.com/MiladShahidi/Kaggle-Protein-Classification/blob/master/trainer/model.py).