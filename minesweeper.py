'''
This code was written in collaboration with Yunfei Chen, Justin Takamiya, and Dev Sashidhar at NC State University
during an AI Course. 
'''

import random
import os
import time
from single_point import Single_Point


def clear_console():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


class MinesweeperConsoleAI:
    def __init__(self, rows, cols, mines, ui=False):
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.ui = ui
        self.board = [[" " for _ in range(cols)] for _ in range(rows)]
        self.revealed = [[False for _ in range(cols)] for _ in range(rows)]
        self.place_mines()
        self.game_over = False
        self.win = False
        self.probability_calculation = None
        self.single_point = Single_Point(self.rows, self.cols, self.board, self.revealed)

    def place_mines(self):
        mine_positions = random.sample(range(self.rows * self.cols), self.mines)
        for pos in mine_positions:
            r, c = divmod(pos, self.cols)
            self.board[r][c] = "M"
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (
                            0 <= r + dr < self.rows
                            and 0 <= c + dc < self.cols
                            and self.board[r + dr][c + dc] != "M"
                    ):
                        if self.board[r + dr][c + dc] == " ":
                            self.board[r + dr][c + dc] = 1
                        else:
                            self.board[r + dr][c + dc] += 1

    def display_board(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.revealed[r][c]:
                    print(self.board[r][c], end=" ")
                else:
                    print("□", end=" ")
            print()
        print()

    def check_win(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] != "M" and not self.revealed[r][c]:
                    return False
        return True

    def ai_click(self, row, col):
        msg = ""
        if self.game_over or self.win:
            return msg
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return msg
        if self.revealed[row][col]:
            return msg
        self.revealed[row][col] = True
        # self.update_edge_attributes()
        if self.board[row][col] == "M":
            self.game_over = True
            msg = f"Game Over! AI clicked on a mine at ({row}, {col})."
            print(msg)
        #NEW
        else:
            self.single_point.add_knowledge(self.single_point.get_adjacent_cells(row, col), self.board[row][col], (row, col))
            #changed elif to if and put inside an else
            if self.board[row][col] == " ":
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr != 0 or dc != 0:
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                                self.ai_click(nr, nc)
        self.win = self.check_win()
        if self.win:
            msg = "Congratulations! All safe cells have been revealed."
            print(msg)
        return msg

    # This is a method that randomly choose row and col
    def get_random_unrevealed_cell(self):
        unrevealed_cells = [
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if not self.revealed[r][c]
        ]
        return random.choice(unrevealed_cells)

    def start_game(self):
        start_time = time.time() # Begin recording time
        msg = ""
        if not self.ui:
            input("Press Enter to start the game...")
        moves = []
        while not self.game_over and not self.win:
            #OLD CODE:
            #row, col = self.probability_calculation.choose_safest_cell()  # The part when we call our own method
            #NEW CODE
            type = ""
            if self.single_point.has_safes():
                type = "SINGLE POINT"
                row, col = self.single_point.get_safes()
                msg = f"[SINGLE POINT] AI clicking on ({row}, {col})"
            else: #HERE IS WHERE GUESSING STRATEGY GOES
                type = "GUESS"
                #ADD Neural Network guess
                msg = f"[GUESS] AI clicking on ({row}, {col})"
            #END OF NEW CODE
            clear_console()
            print(type)
            print(msg)
            if self.ui:
                yield msg
            moves.append((row, col))
            msg = self.ai_click(row, col)
            if self.ui:
                if "Congratulations!" in msg or "Game Over!" in msg: # If the message contains Congratulations or Game Over after the click,
                    msg += f"(Time taken: {time.time() - start_time})" # Add the time taken to the message
                yield msg
            self.display_board()
            print(self.single_point.safes)
            for i in moves:
                print(i)
            if self.game_over:
                print(self.single_point.kb)
            if self.win:
                break
            if not self.game_over and not self.win:
                time.sleep(0)  # Adjust for time during the decision


if __name__ == "__main__":
    game = MinesweeperConsoleAI(20, 20, 45)  # Change the row, col and mine numbers here
    game.start_game()