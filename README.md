# CatsNDogs-TAI-2024
Repository for the "cats and dogs" project for the course Techniques of Artificial Intelligence (2024)

The full details of our model training and hyperparameters optimization is given in the notebook `training.ipynb`. 

For ease of execution, we provide three scripts that allow the user to go from the initial dataset of images to the ready-to-use prediction GUI, by executing the following:
1. The VGG16 (1000) features extraction for each image of the training dataset is conducted by executing `python extract_features.py`. This will output labels and features files.
2. Our classification model can be trained by executing `python train.py`. This will output a (compressed) file, `final_model`, which contains all the parameters for the pre-trained model.
3. Finally, one must execute `python main.py` in order to proceed to predictions. 
