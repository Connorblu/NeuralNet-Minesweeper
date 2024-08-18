First, ensure that torch has been installed on your system or in your environment. "pip3 install torch" will work
for most systems.

In order to run tests on the neural network, you can execute test_nn.py with the command "python test_nn.py". 
If line 131 has game.testing_nn = False, the program will run with random guessing. If this is set to True, 
the file will run the neural network. This can be used to compare guessing and neural network performance
as seen in the project report.

Training the model we uesd can be done by running "python train_model.py", and the model will be saved to the models folder.

Training the CNN model can be done by running "python3 cnn_train_model.py" and the model will be saved to the models folder.