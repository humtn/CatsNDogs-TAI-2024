# CatsNDogs-TAI-2024
Repository for the "cats and dogs" project for the course Techniques of Artificial Intelligence (2024).

Our project brings the following three improvements/changes from the past implementation:
1. the project was ported from `tensorflow` to `pytorch`, 
2. we now use a bigger dataset of images,
3. the feature extractor relies on the VGG16 pre-trained model.

This repository is composed of :
- `training.ipynb`, a python notebook that contains the full detail of our model training and hyperparameter optimization procedure, as well as a glimpse at the missclassified images,
- `train.py`, a python script that upon execution trains the model using the optimal hyperparameter on the full dataset, and outputs a (compressed) file containing the weights for our model,
- `main.py`, a python script that starts a GUI from which the user can make predictions. **This file can be ran as is, as we already provide the pre-trained model in this repository**.

Additionally, if one wants to re-run the feature extraction on the input dataset (those features are provided within this repository), one must download the training dataset (not included in this repository for storage reasons) from [Kaggle](https://www.kaggle.com/datasets/chetankv/dogs-cats-images/data) and place it in the `dataset` folder (splitting the training and testing accordingly to `/test` and `/train`). The feature extraction component is located within `training.ipynb` and will output the files containing the features in `/features`.

The folder `/images_to_predict` contains some basic images to test our classifier.