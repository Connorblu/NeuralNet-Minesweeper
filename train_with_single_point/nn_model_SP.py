import torch
import torch.nn as nn
import torch.nn.functional as f

class Minesweeper_20x20_Model_SP(nn.Module):

    '''
    Im thinking we need a 20 x 20 x 2 layer for the input, because the minesweeper 
    board for this model is 20 x 20 and we need 1 layer for the values and 1 layer that
    shows what has been revealed so far. This will be a linear model, so it will not have 2D structure
    '''
    def __init__(self, in_features=800, h = 600, out_features=400):
        super().__init__()
        #Create a fully connected layer, that goes from in_features to hidden 1 (h1)
        self.fc1 = nn.Linear(in_features, h)

        self.fc2 = nn.Linear(h, h)

        self.fc3 = nn.Linear(h, h)

        self.fc4 = nn.Linear(h, h)

        self.fc5 = nn.Linear(h, h)

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
        x = self.out(x)
        return x
    

class CNN_Minesweeper(nn.Module):

    def __init__(self, rows, cols):
        super(CNN_Minesweeper, self).__init__()
        self.rows = rows
        self.cols = cols

        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)

        self.pool = nn.MaxPool2d(2, 2)

        reduced_size = (rows // 2 // 2) * (cols // 2 // 2) * 128

        self.fc1 = nn.Linear(reduced_size, 512)
        self.fc2 = nn.Linear(512, rows * cols)

    def forward(self, x):
        x = f.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(x)
        #print("Post-pool 1 shape: ", x.shape)
        x = f.leaky_relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(x)

        x = f.leaky_relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        #print("Post-pool 2 shape: ", x.shape)
        x = x.view(-1, self.num_flat_features(x))
        #print("Flattened shape: ", x.shape)
        x = f.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        return torch.sigmoid(x)
    
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