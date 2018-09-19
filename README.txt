An Image Classifier project that I built utilizing transfer machine learning from a VGG16 or VGG19 model
(pretrained on ImageNet). The program loads a choice of one of the two models and replaces the classifier 
on the model and trains that portion alone using the train.py file. The program prints out status updates 
during training and compares training loss to validation loss and accuracy to help determine a good stopping 
point for training. After the model is trained, the program runs it on a separate test data to verify the 
accuracy of the model. Then, the model has a checkpoint saved with a utility function from the utils.py file.

The predict.py file can then be run which will load the checkpoint file from the train.py function, and then 
use it to attempt to classify a given flower image supplied by the user. The function will output a formatted 
graphic of the input flower picture, accompanied by a bar chart depicting the top-K most likely flower species 
that are in the picture with associated probability values.

Note: train data was too large to upload to this repository, but can be provided upon request.