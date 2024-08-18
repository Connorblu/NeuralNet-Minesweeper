import torch
import sys
import nn_model

class NN_Player:
    def __init__(self, row, column, model_path):
        self.model = nn_model.Minesweeper_20x20_Model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.rows = row
        self.cols = column
    
    def move_from_pred(self, pred, revealed):
        min_index = None
        min_value = sys.float_info.max
        for r in range(self.rows):
            for c in range(self.cols):
                if revealed[r][c] == 0 and pred[r * self.cols + c].item() < min_value:
                    min_index = (r,c)
                    min_value = pred[r * self.cols + c].item()
        return min_index
    
    def make_move(self, board, revealed):
        visible_board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for row in range(self.rows):
            for col in range(self.cols):
                if revealed[row][col]:
                    if board[row][col] == " ":
                        visible_board[row][col] = 0
                    else:
                        visible_board[row][col] = board[row][col]
        
        revealed_binary = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for row in range(self.rows):
            for col in range(self.cols):
                if revealed[row][col]:
                    revealed_binary[row][col] = 1
        
        input = visible_board + revealed_binary
        x = torch.Tensor(input)
        x = x.reshape(2 * self.rows * self.cols)

        #Gets a prediction from the neural network model
        y_pred = self.model.forward(x)

        return self.move_from_pred(y_pred, revealed)

