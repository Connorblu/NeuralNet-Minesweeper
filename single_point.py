
class Single_Point():
    
    def __init__(self, rows, cols, board, revealed):
        self.rows = rows
        self.cols = cols
        self.board = board
        self.revealed = revealed
        self.flag = False
        self.found = set()
        self.mines = set()
        self.safes = set()
        self.already_found = set()
        self.kb = set()
    

    def add_knowledge(self, seen_cells, num_mines, loc):
        #First, update the knowledgebase to recognize that we have opened this cell
        self.already_found.add(loc)
        self.revealed[loc[0]][loc[1]] = True
        to_remove = set()
        new_knowledge = set()
        f_zero = Fact({loc}, 0)
        for el in self.kb:
            if f_zero.subset(el):
                to_remove.add(el)
                new_knowledge.add(el.subtract(f_zero))
        self.kb = self.kb.difference(to_remove)
        self.kb = self.kb.union(new_knowledge)

        #If this is a 0 cell, we have no knew knowledge and therefore can exit the function
        if seen_cells != 0:
            #now we know that this cell contains a number, and therefore has valuable information
            f = Fact(seen_cells, num_mines)
            self.kb.add(f)
            #Case where we can mark all seen_cells as mines
            if len(seen_cells) == num_mines:
                for cell in seen_cells:
                    self.mines.add(cell)
                
                #reduction
                to_remove = set()
                new_knowledge = set()
                for el in self.kb:
                    if f_zero.subset(el):
                        to_remove(el)
                        new_knowledge.add(el.subtract(f))
                self.kb = self.kb.difference(to_remove)
                self.kb = self.kb.union(new_knowledge)
        
        new_knowledge = set()
        for el1 in self.kb:
            #Let's add any safe spaces we are sure of
            if el1.value == 0:
                for cell in el1.cells:
                    if cell not in self.already_found:
                        self.safes.add(cell)
            #Run inference to see if we can find any more safe cells
            for el2 in self.kb:
                if el1 != el2 and (el1.subset(el2) or el2.subset(el1)):
                    new_knowledge.add(el1.subtract(el2))
        
        for k in new_knowledge:
            self.kb.add(k)
            if k.value == 0:
                for cell in k.cells:
                    if cell not in self.already_found:
                        self.safes.add(cell)
        
        #CLEAN THE KB:
        to_remove = set()
        for el in self.kb:
            if len(el.cells) == 0:
                to_remove.add(el)
        self.kb = self.kb.difference(to_remove)

        self.safes = self.safes - self.already_found
    
    def get_safes(self):
        if not self.flag:
            self.flag = True
            return (0,0)
        return self.safes.pop()
    
    def has_safes(self):
        return len(self.safes) > 0

    def get_adjacent_cells(self, i, j):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return {(i + dr, j + dc) for dr, dc in directions if 0 <= i + dr < self.rows and 0 <= j + dc < self.cols and self.revealed[i + dr][j + dc] == False}


class Fact():

    def __init__(self, cells, value):
        self.cells = cells
        if(value == ' '):
            self.value = 0
        else:
            self.value = int(value)
    
    def __eq__(self, other_fact):
        if isinstance(other_fact, Fact) and self.cells == other_fact.cells and self.value == other_fact.value:
            return True
        return False
    
    def __str__(self):
        return f'({str(self.cells)}, {self.value})'

    def __hash__(self):
        return hash(str(self))
    
    def subset(self, other_fact):
        return self.cells <= other_fact.cells
    
    #Taking the intersection of two Facts can lead to new insights.
    def subtract(self, other_fact):
        if len(self.cells - other_fact.cells) != 0:
            return Fact(self.cells - other_fact.cells, abs(self.value - other_fact.value))
        else:
            return Fact(other_fact.cells - self.cells, abs(self.value - other_fact.value))
