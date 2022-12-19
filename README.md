# AI-and-ML<br>
1.Write a Program to Implement Breadth First Search using Python.<br>
graph = {<br><br>
 '1' : ['2','10'],<br>
 '2' : ['3','8'],<br>
 '3' : ['4'],<br>
 '4' : ['5','6','7'],<br>
 '5' : [],<br>
 '6' : [],<br>
 '7' : [],<br>
 '8' : ['9'],<br>
 '9' : [],<br>
 '10' : []<br>
 }<br>
visited = []<br>
queue = []<br>
def bfs(visited, graph, node):<br>
    visited.append(node)<br>
    queue.append(node)<br>
    while queue:<br>
        m = queue.pop(0)<br>
        print (m, end = " ")<br>
        for neighbour in graph[m]:<br>
            if neighbour not in visited:<br>
                visited.append(neighbour)<br>
                queue.append(neighbour)<br>
print("Following is the Breadth-First Search")<br>
bfs(visited, graph, '1')<br>

OUTPUT:<br>
Following is the Breadth-First Search<br>
1 2 10 3 8 4 9 5 6 7 <br>
************************************************************************************************
2.Write a Program to Implement Best First Search using Python.<br>
graph = {<br>
    '5': ['3','7'],<br>
    '3': ['2','4'],<br>
    '7': ['6'],<br>
    '6':[],<br>
    '2': ['1'],<br>
    '1':[],<br>
    '4': ['8'],<br>
    '8':[]<br>
}<br>
visited = set() # Set to keep track of visited nodes of graph.<br>
def dfs(visited, graph, node): #function for dfs<br>
    if node not in visited:<br>
        print (node)<br>
        visited.add(node)<br>
        for neighbour in graph[node]:<br>
            dfs(visited, graph, neighbour)<br>
    # Driver Code<br>
print("Following is the Depth-First Search")<br>
dfs(visited, graph, '5')<br>

OUTPUT:<br>
Following is the Depth-First Search<br>
5<br>
3<br>
2<br>
1<br>
4<br>
8<br>
7<br>
6<br>
************************************************************************************
3. Write a Program to Implement Depth First Search using Python.<br>
        # Using a Python dictionary to act as an adjacency list<br>
graph = {<br>
'5' : ['3','7'],<br>
'3' : ['2', '4'],<br>
'7' : ['6'],<br>
'6': [],<br>
'2' : ['1'],<br>
'1':[],<br>
'4' : ['8'],<br>
'8' : []<br>
}<br>
visited = set() # Set to keep track of visited nodes of graph.<br>
def dfs(visited, graph, node): #function for dfs<br>
  if node not in visited:<br>
    print (node)<br>
    visited.add(node)<br>
    for neighbour in graph[node]:<br>
      dfs(visited, graph, neighbour)<br>

           # Driver Code<br>
print("Following is the Depth-First Search")<br>
dfs(visited, graph, '5')<br>

OUTPUT:<br>
Following is the Depth-First Search<br>
5<br>
3<br>
2<br>
1<br>
4<br>
8<br>
7<br>
6<br>
*******************************************************************************************************
4. Write a Program to Implement Tic-Tac-Toe application using Python.<br>
import numpy as np<br>
import random<br>
from time import sleep<br>

def create_board():<br>
    return(np.array([[0, 0, 0],<br>
                    [0, 0, 0],<br>
                    [0, 0, 0]]))<br>

def possibilities(board):<br>
    l = []<br>

    for i in range(len(board)):<br>
        for j in range(len(board)):<br>

            if board[i][j] == 0:<br>
                l.append((i, j))<br>
    return(l)<br>

def random_place(board, player):<br>
    selection = possibilities(board)<br>
    current_loc = random.choice(selection)<br>
    board[current_loc] = player<br>
    return(board)<br>


def row_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True<br>

        for y in range(len(board)):<br>
            if board[x, y] != player:<br>
                win = False<br>
                continue<br>

        if win == True:<br>
            return(win)<br>
    return(win)<br>

def col_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True<br>
    
        for y in range(len(board)):<br>
            if board[y][x] != player:<br>
                win = False<br>
                continue<br>

        if win == True:<br>
            return(win)<br>
    return(win)<br>


def diag_win(board, player):<br>
    win = True<br>
    y = 0<br><br>
    for x in range(len(board)):<br>
        if board[x, x] != player<br>:
            win = False<br>
        
    if win:<br>
        return win<br>
    win = True<br>
    if win:<br>
        for x in range(len(board)):<br>
            y = len(board) - 1 - x<br>
            if board[x, y] != player:<br>
                win = False<br>
    return win<br>

def evaluate(board):<br>
    winner = 0<br>

    for player in [1, 2]:<br>
        if (row_win(board, player) or<br>
                col_win(board,player) or<br>
                diag_win(board,player)):<br>
            winner = player<br>
            
    if np.all(board != 0) and winner == 0:<br>
        winner = -1<br>
    return winner<br>

def play_game():<br>
    board, winner, counter = create_board(), 0, 1<br>
    print(board)<br>
    sleep(2)<br>

    while winner == 0:<br>
        for player in [1, 2]:<br>
            board = random_place(board, player)<br>
            print("Board after " + str(counter) + " move")<br>
            print(board)<br>
            sleep(2)<br>
            counter += 1<br>
            winner = evaluate(board)<br>
            if winner != 0:<br>
                break<br>
    return(winner)<br>

print("Winner is: " + str(play_game()))<br>

OUTPUT:<br>

[[0 0 0]<br>
 [0 0 0]<br>
 [0 0 0]]<br>
Board after 1 move<br>
[[0 0 0]<br>
 [0 0 0]<br>
 [1 0 0]]<br>
Board after 2 move<br>
[[0 0 2]<br>
 [0 0 0]<br>
 [1 0 0]]<br>
Board after 3 move<br>
[[0 0 2]<br>
 [0 0 0]<br>
 [1 1 0]]<br>
Board after 4 move<br>
[[0 0 2]<br>
 [0 2 0]<br>
 [1 1 0]]<br>
Board after 5 move<br>
[[0 0 2]<br>
 [1 2 0]<br>
 [1 1 0]]<br>
Board after 6 move<br><br>
[[0 2 2]<br>
 [1 2 0]<br>
 [1 1 0]]<br>
Board after 7 move<br>
[[0 2 2]<br>
 [1 2 1]<br>
 [1 1 0]]<br>
Board after 8 move<br>
[[2 2 2]<br>
 [1 2 1]<br>
 [1 1 0]]<br>
Winner is: 2<br>
