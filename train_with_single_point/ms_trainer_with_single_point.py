import random
import os
import sys





class MinesweeperTrainerAI:
    def __init__(self, rows, cols, mines):
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.revealed = [[0 for _ in range(cols)] for _ in range(rows)]
        self.visible_board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.mine_placements = [[0 for _ in range(cols)] for _ in range(rows)]
        self.single_point = None
        self.place_mines()
        self.game_over = False
        self.win = False
    
    def init_single_point(self, sp):
        self.single_point = sp

    def place_mines(self):
        mine_positions = random.sample(range(self.rows * self.cols), self.mines)
        for pos in mine_positions:
            r, c = divmod(pos, self.cols)
            self.board[r][c] = "M"
            self.mine_placements[r][c] = 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (
                            0 <= r + dr < self.rows
                            and 0 <= c + dc < self.cols
                            and self.board[r + dr][c + dc] != "M"
                    ):
                        if self.board[r + dr][c + dc] == 0:
                            self.board[r + dr][c + dc] = 1
                        else:
                            self.board[r + dr][c + dc] += 1


    def check_win(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] != "M" and not self.revealed[r][c] == 1:
                    return False
        return True

    def ai_click(self, row, col):
        if self.game_over or self.win:
            return
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return
        if self.revealed[row][col] == 1:
            print(f"CLICKED ON AN ALREADY CLICKED TILE ROW={row} and COL={col}")
            return
        self.revealed[row][col] = 1
        # self.update_edge_attributes()
        if self.board[row][col] == "M":
            self.game_over = True
        #NEW
        else:
            self.single_point.add_knowledge(self.single_point.get_adjacent_cells(row, col), self.board[row][col], (row, col))
            self.visible_board[row][col] = self.board[row][col]
            #changed elif to if and put inside an else
            if self.board[row][col] == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr != 0 or dc != 0:
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.revealed[nr][nc] != 1:
                                self.ai_click(nr, nc)
        self.win = self.check_win()

    # This is a method that randomly choose row and col
    def get_random_unrevealed_cell(self):
        unrevealed_cells = [
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if not self.revealed[r][c] == 1
        ]
        return random.choice(unrevealed_cells)



    def start_game(self):
        #Reset board for trainer
        self.board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.revealed = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.visible_board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.mine_placements = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.place_mines()
        self.game_over = False
        self.win = False
    
    def move_from_pred(self, pred):
        min_index = None
        min_value = sys.float_info.max
        for r in range(self.rows):
            for c in range(self.cols):
                if self.revealed[r][c] == 0 and pred[r * self.cols + c].item() < min_value:
                    min_index = (r,c)
                    min_value = pred[r * self.cols + c].item()
        if not min_index:
            return self.get_random_unrevealed_cell()
        return min_index
    
    def get_revealed(self):
        return self.revealed
    
    def get_board(self):
        return self.board
    
    def get_visible_board(self):
        return self.visible_board
    
    def get_mine_arrangements(self):
        return self.mine_placements


