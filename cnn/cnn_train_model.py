import nn_model
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import cnn_ms_trainer
import sys
import time
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

def cnn_train_model():
    t1 = time.perf_counter()
    #Change this if you want to train a different model
    rows, cols = 20, 20
    model = nn_model.CNN_Minesweeper(rows, cols)

    #Third param is number of mines
    game = cnn_ms_trainer.MinesweeperTrainerAI(model.rows, model.cols, 40)

    #Choosing our loss function
    criterion = torch.nn.BCELoss()

    #Time on cuda: 39 seconds
    #Time on cpu: 88 seconds
    device = get_default_device()
    print("Training on " + str(device))

    model = to_device(model, device)
    print("Model initialized and moved to: ", device)

    #Note: Lower lr (learning rate) means longer training times
    initial_lr = 0.3
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    print("Optimizer and Scheduler setup with initial LR: ", optimizer.param_groups[0]['lr'])

    #Choose number of games to play here
    epochs = 1000
    games_per_epoch = 100
    total_games = epochs * games_per_epoch
    #Keeping track of how many games the nn wins during training
    games_won = 0
    games_played = 0
    for epoch in range(epochs):
        total_loss = 0
        for game_index in range(games_per_epoch):
        #Create our "dataset" with numpy arrays by playing the game
            game.start_game()
            #print(f"Starting Epoch {epoch}/{epochs}, Game {game_index + 1}/{games_per_epoch}")

            worst_loss = float('-inf')
            best_loss = float('inf')

            #Playing minesweeper until we lose
            while game.win == False and game.game_over == False:

                visible_board = game.visible_board
                revealed = game.revealed
                input_tensor = nn_model.prepare_input(visible_board, revealed, model.rows, model.cols)
                input_tensor = to_device(input_tensor, device)



                #Gets a prediction from the neural network model
                y_pred = model(input_tensor)
                #print(f"Post forward shape: {y_pred.shape}")
                y_pred = y_pred.squeeze(1)
                #print("Predicted probabilities by model: \n", y_pred.detach().cpu().numpy().reshape(model.rows, model.cols))
            
                y_pred_for_move = y_pred.squeeze()
                move = game.move_from_pred(y_pred_for_move)
                game.ai_click(move[0], move[1])
                #print(f"Move chosen: {move}, Cell value: {'Mine' if game.board[move[0]][move[1]] == 'M' else 'Safe'}")

                #Gets the mine arrangment so that we can compare to the predicted mine arrangement
                np_mine_arrangment = np.reshape(np.array(game.mine_placements), model.rows * model.cols)
                tensor_mines = to_device(torch.Tensor(np_mine_arrangment), device)
                tensor_mines = tensor_mines.unsqueeze(0)

            
                #print(f"tensor_mines shape: {tensor_mines.shape}")
                #Calculate loss and backpropagate
                loss = criterion(y_pred, tensor_mines)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                current_loss = loss.item()
                worst_loss = max(current_loss, worst_loss)
                best_loss = min(current_loss, best_loss)


            if game.win:
                games_won += 1
            
            avg_loss = total_loss / (games_per_epoch)
            scheduler.step(avg_loss)

            
            #print(f"End of Game {game_index + 1}: Learning Rate: {current_lr}, Current Loss: {current_loss}")
        
        #if game.win == True:
             #games_won += 1
        if (epoch % 100 == 0):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch} , Learning Rate: {current_lr}, Games played: {(epoch + 1) * games_per_epoch}, Wins: {games_won}, best loss:{best_loss}, avg loss:{avg_loss}")
    
    t2 = time.perf_counter()
    #win_rate = (games_won / games_played) * 100 
    print(f"Model is about to save! It won {games_won} games out of {total_games} and took {t2 - t1} seconds!")
    save_model(model, 'models/20x20_Model1')

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(el, device) for el in data]
    return data.to(device, non_blocking=True)

if __name__ == "__main__":
    print("About to train!")
    cnn_train_model()

