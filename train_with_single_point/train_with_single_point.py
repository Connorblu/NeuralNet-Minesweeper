import nn_model_SP
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import ms_trainer_with_single_point
import sys
from single_point_trainer import Single_Point
import time

def train_model():
    t1 = time.perf_counter()
    #Change this if you want to train a different model
    model = nn_model_SP.Minesweeper_20x20_Model_SP()

    #Third param is number of mines
    game = ms_trainer_with_single_point.MinesweeperTrainerAI(model.rowdim,model.coldim,40)


    #Choosing our loss function
    criterion = torch.nn.MSELoss()

    device = get_default_device()
    print("Training on " + str(device))

    model = to_device(model, device)

    #Note: Lower lr (learning rate) means longer training times
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

    #Choose number of games to play here
    epochs = 100000

    #Keeping track of how many games the nn wins during training
    games_won = 0
    guesses_allowed = 10
    for i in range(epochs):
        #Create our "dataset" with numpy arrays by playing the game
        game.start_game()
        single_point = Single_Point(model.rowdim, model.coldim, game.board, game.revealed)
        game.init_single_point(single_point)

        #Gets the mine arrangment so that we can compare to the predicted mine arrangement
        np_mine_arrangment = np.reshape(np.array(game.get_mine_arrangements()), model.rowdim * model.coldim)
        tensor_mines = to_device(torch.Tensor(np_mine_arrangment), device)

        worst_loss = sys.float_info.min
        best_loss = sys.float_info.max

        num_guesses = guesses_allowed
        games_won_with_single_point = 0
        games_won_with_nn = 0
        used_nn = False
        #Playing minesweeper until we lose
        while game.win == False and game.game_over == False:

            if single_point.has_safes():
                loc = single_point.get_safes()
                game.ai_click(loc[0], loc[1])
            elif num_guesses > 0:

                #Input layer consists of the visible board and what cells are revealed
                input = game.get_visible_board() + game.get_revealed()
                x = to_device(torch.Tensor(input), device)
                x = x.reshape(2 * model.rowdim * model.coldim)

                #Gets a prediction from the neural network model
                y_pred = model.forward(x)

                if i % 10000 == 0 and used_nn == False:
                    print("PREDICTED")
                    for row in range(model.rowdim):
                        for col in range(model.coldim):
                            print(y_pred[row * model.coldim + col].item(), end=" ")
                        print()
                    print("ACTUAL")
                    for row in range(model.rowdim):
                        for col in range(model.coldim):
                            print(game.get_visible_board()[row][col], end=" ")
                        print()
                    print("BOARD")
                    for row in range(model.rowdim):
                        for col in range(model.coldim):
                            print(game.get_board()[row][col], end=" ")
                        print()
                    print("REVEALED")
                    for row in range(model.rowdim):
                        for col in range(model.coldim):
                            print(game.get_revealed()[row][col], end=" ")
                        print()


                used_nn = True
                move = game.move_from_pred(y_pred)
                game.ai_click(move[0], move[1])

                #Calculate loss and backpropagate
                loss = criterion(y_pred, tensor_mines)
                worst_loss = max(loss, worst_loss)
                best_loss = min(loss, best_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_guesses -= 1
            else:
                break
        
        if game.win == True:
            if used_nn == True:
                 games_won_with_nn += 1
            else:
                games_won_with_single_point += 1
            games_won += 1
        if (i % 10000 == 0):
            print(f"Epoch: {i} , best loss:{best_loss}, worst loss:{worst_loss}")
            guesses_allowed += 4
    
    t2 = time.perf_counter()
    print(f"Model is about to save! It won {games_won} games while training, and took {t2 - t1} seconds!")
    print(f"{games_won_with_single_point} games were won with only single point, and {games_won_with_nn} were won with the help of the neural net!")
    torch.save(model.state_dict(), '20x20_Model_with_SP')

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(el, device) for el in data]
    return data.to(device, non_blocking=True)

if __name__ == "__main__":
    print("About to train!")
    train_model()

