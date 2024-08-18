

def get_adjacent_cells(self, i, j):
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    return [(i + dr, j + dc) for dr, dc in directions if 0 <= i + dr < self.rows and 0 <= j + dc < self.cols]

def update_edge_attributes(self):
    for i in range(self.rows):
        for j in range(self.cols):
            if self.revealed[i][j]:
                self.cell_properties[i][j].update({"open": True, "edge": False, "edgeCount": 0})
                adjacent_cells = self.get_adjacent_cells(i, j)
                for nr, nc in adjacent_cells:
                    if not self.revealed[nr][nc]:
                        # Each unrevealed neighbor of a revealed cell might be an edge cell
                        self.cell_properties[nr][nc]["edge"] = True
                        # Increment the edge count of the revealed cell to indicate potential mines around it
                        self.cell_properties[nr][nc]["edgeCount"] += 1
            else:
                # This cell is not revealed; update its open and edge status based on surrounding context
                adjacent_cells = self.get_adjacent_cells(i, j)
                edge_count = sum(1 for nr, nc in adjacent_cells if self.revealed[nr][nc])
                self.cell_properties[i][j].update({"open": False, "edge": edge_count > 0, "edgeCount": edge_count})