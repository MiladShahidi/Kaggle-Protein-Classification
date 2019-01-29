# Image Classification with Deep Convolutional Neural Networks in TensorFlow

<img src="images/proteins.png" align="right" width="256">

In this project, I use data from [Kaggle's Human Protein Classification competition](https://www.kaggle.com/c/human-protein-atlas-image-classification) and train a deep convolutional neural network to identify various types of proteins in microscopic images. The model achieves an F1 score of 0.471 on the test set, which is **in the top 15% of the final leaderboard** of the competition.

Since each image may contain more than one type of protein (28 in total), this is a multi-class multi-label classification problem. One of the challenges specific to multi-label classification problems, is splitting the data into balanced training and validation sets. I use **iterative splitting** (Sechidis, Tsoumakas, and Vlahavas 2011), which is specifically designed for multilabel data, to ensure that training and validation sets have similar distributions.

I use random flipping and rotation to augment the training data. My experiments indicate that this improves the performance of the model significantly. To address the problem of imbalance in the data, I optimaize the classification threshold for each class by searching for the empirically optimal threshold in the validation set.

The dataset takes up **more than 30GB** and is **too large to fit in memory**. To efficiently handle this large dataset, I design a data pipeline that gradually loads minibathces from a Google Cloud Storage bucket during training.

<img src="images/plots.png" align="center">

Given that training multiple models for hyperparameter search is computatinally expensive, I **manually tuned the learning rate** during training, lowering it whenever the validation loss stopped improving. The graphs above show training and validation losses along with macor F1 Score. F1 score combines precision and recall and is a more suitable performace metric than accuracy, in problems with class imbalanced.

Training was done on a single Tesla P100 GPU on Google Cloud ML Engine. The complete TensorFlow training code is available [here](https://github.com/MiladShahidi/Kaggle-Protein-Classification/blob/master/trainer/model.py).

## References

Sechidis, Konstantinos, Grigorios Tsoumakas, and Ioannis Vlahavas. 2011. “On the stratification of multi-label data”. In *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*, 145–158. Springer.