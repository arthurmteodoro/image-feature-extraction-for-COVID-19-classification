# xDNN-Experiments
Use of several Convolutional Neural Networks (CNN) as a feature extractor for the xDNN algorithms.

# Dependences
This project uses the following dependencies:

* numpy
* scipy
* scikit-learn
* tensorflow
* tqdm

To install all dependencies, use:

`$ pip install -r requirements.txt`

# How to run

To execute the code, use the follow command

`$ python main.py --data-dir <DATASET_PATH> -- model <MODEL>`

where:

* DATASET_PATH is a path to dataset
* MODEL is a CNN model to use as Feature Extractor

Example:

`python main.py --data-dir ../covid-ct2 --model efficientnetb0`

Available params in main file:

* **data-dir**: path to dataset (contain all images if validation-dir not specified)
* **model**: model to use as feature extractor
* **validation-dir**: validation directory path used
* **validation-split**: Percentage of the dataset specified in data-dir will be used for validation (used only if validation-dir is not specified)

# Available Models

This follow models is available to use as feature extractor:

* VGG16 (`--model vgg16`)
* EfficientNetB0 (`--model efficientnetb0`)