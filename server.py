from bottle import Bottle
from minesweeper import MinesweeperConsoleAI

ROWS = 20
COLS = 20
MINES = 45
DISPLAY_UNIT = 30

# Tableau 10 Color Palette Hex Codes
# https://gist.github.com/leblancfg/b145a966108be05b4a387789c4f9f474
TABLEAU10 = [
    "#5778a4",  # blue:
    "#e49444",  # orange:
    "#d1615d",  # red:
    "#85b6b2",  # teal:
    "#6a9f58",  # green:
    "#e7ca60",  # yellow:
    "#a87c9f",  # purple:
    "#f1a2a9",  # pink:
    "#967662",  # brown:
    "#b8b0ac",  # grey:
]

def header(reload_time):
    HEADER = """
<header>
"""
    if reload_time < 100:
        HEADER += f'<meta http-equiv="refresh" content="{reload_time}">'
    HEADER += f"""
<style>
* {{
    font-family: "Lucida Console", "Courier New", monospace;
    color: #333333;
}}
h1 {{
    font-size: 32px;
    margin-left: 110px; 
    float: left;
}}
.container {{
    width: 100%; 
    overflow: auto;
}}
.board{{
    width: {(DISPLAY_UNIT + 4) * COLS}px;
    height: {(DISPLAY_UNIT + 4) * ROWS}px;
    padding: 5px;
    margin-top: 10px;
    margin-bottom: 10px;
    margin-right: 30 px;
    margin-left: 100px; 
    float: left;
}}
.status{{
    width: {(DISPLAY_UNIT + 4) * COLS / 2 + 200}px;
    height: {(DISPLAY_UNIT + 4) * ROWS}px;
    padding: 10px;
    margin-top: 10px;
    margin-bottom: 10px;
    margin-right: 30 px;
    margin-left:  {(DISPLAY_UNIT + 4) * COLS + 150}px;
    font-size: 20px;
    border-style: solid;
    border-color: #b8b0ac;
    border-radius: 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column-reverse;    
}}
input {{
    width: {(DISPLAY_UNIT + 4) * COLS / 2 + 220}px;
    height: 30px;
    font-size: 20px;
    border-style: solid;
    border-color: #b8b0ac;
    border-radius: 20px;
    margin-top: 10px;
    margin-bottom: 10px;
    margin-right: 30 px;
    margin-left:  {(DISPLAY_UNIT + 4) * COLS + 150}px;

}}
.tile{{
    border: 2px solid;
    border-radius: 5px;
    font-size: 32px;
    text-align: center;
    float: left;
    width: {DISPLAY_UNIT}px;
    height: {DISPLAY_UNIT}px;
}}
"""

    for i,rgb in enumerate(TABLEAU10):
        HEADER += f"""
.c{i} {{
    color: white;
    background-color: {rgb};
    border-color: white;
}}
"""

    HEADER += """</style>
</header>
<h1>combined</h1>

"""
    return HEADER

class MinesweeperWebUI(Bottle):
    def __init__(self, name):
        super(MinesweeperWebUI, self).__init__()
        self.name = name
        self.route('/', callback=self.index)
        self.msg = ""

    def new_game(self):
        self.game = MinesweeperConsoleAI(ROWS,  COLS, MINES, ui=True)  # Change the row, col and mine numbers here
        self.gen = self.game.start_game()
        self.msg = ""

    def html(self, reload_time):
        s = "<html>\n"
        s += header(reload_time)
        s += '<div class="container id="containerdiv">'
        s += '<div class="board">'
        for r in range(ROWS):
            for c in range(COLS):
                if self.game.revealed[r][c]:
                    ch = self.game.board[r][c]
                    if ch == " ":
                        ch = "&nbsp;"
                        cls = "9"
                    else:
                        cls = str(ch)
                    # print(f"'{ch}'")
                else:
                    ch = " "
                    cls = "0"
                s += f'<div class=" c{cls} tile">{ch}</div>'
            s+= '</br>\n'
        s += f"""</div>
<div class="status">
{self.msg}
</div>
</div>
<form action="/">
    <input type="submit" value="Start New Board" />
</form>
</html>
"""
        return s

    def index(self):
        reload_time = 0
        try:
            msg = next(self.gen)
            if msg:
                self.msg += msg + "<br/>"
                if "Congratulations!" in msg or "Game Over!" in msg:
                    reload_time = 100

        except Exception as e:
            print(e)
            self.new_game()
        return self.html(reload_time)


app = MinesweeperWebUI('MinesweeperWebUI')
app.run(host='localhost', port=8080)