import torch
import torch.nn as nn
import torch.nn.functional as f

class Minesweeper_20x20_Model(nn.Module):

    '''
    Im thinking we need a 20 x 20 x 2 layer for the input, because the minesweeper 
    board for this model is 20 x 20 and we need 1 layer for the values and 1 layer that
    shows what has been revealed so far. This will be a linear model, so it will not have 2D structure

    h = 1600 for Model 7
    h = 400 for Model 6
    '''
    def __init__(self, in_features=800, h=400, out_features=400):
        super().__init__()
        #Create a fully connected layer, that goes from in_features to hidden 1 (h1)
        self.fc1 = nn.Linear(in_features, h)

        self.fc2 = nn.Linear(h, h)

        self.fc3 = nn.Linear(h, h)

        self.fc4 = nn.Linear(h, h)

        self.fc5 = nn.Linear(h, h)

        self.fc6 = nn.Linear(h, h)

        #self.fc7 = nn.Linear(h, h)

        #self.fc8 = nn.Linear(h, h)

        self.out = nn.Linear(h, out_features)

        self.rowdim = 20
        self.coldim = 20

    '''
    This is the function that moves data forward through the network.
    We will use ReLU because of efficiency, but I think sigmoid might be better 
    for the actual problem. If training does not take too long, I will try 
    sigmoid and see if we get any better performance.
    '''
    def forward(self, x):
        x = f.sigmoid(self.fc1(x))
        x = f.sigmoid(self.fc2(x))
        x = f.sigmoid(self.fc3(x))
        x = f.sigmoid(self.fc4(x))
        x = f.sigmoid(self.fc5(x))
        x = f.sigmoid(self.fc6(x))
        #x = f.sigmoid(self.fc7(x))
        #x = f.sigmoid(self.fc8(x))
        x = self.out(x)
        return x
    

class CNN_Minesweeper(nn.Module):

    def __init__(self, rows, cols):
        super(CNN_Minesweeper, self).__init__()
        self.rows = rows
        self.cols = cols

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 9, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def prepare_input(visible_board, revealed, rows, cols):
    input_tensor = torch.tensor([visible_board, revealed], dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor.view(1, 2, rows, cols)
