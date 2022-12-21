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
2.Write a Program to Implement Depth First Search using Python..<br>
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
3. Write a Program to Implement Tic-Tac-Toe application using Python.<br>
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
***********************************************************************************************
4.Write a Program to Implement Best First Search using Python.<br>
      #Write a Program to Implement Best First Search using Python.<br>
from queue import PriorityQueue<br>
import matplotlib.pyplot as plt<br>
import networkx as nx<br>

     # for implementing BFS | returns path having lowest cost<br>
def best_first_search(source, target, n):<br>
    visited = [0] * n<br>
    visited[source] = True<br>
    pq = PriorityQueue()<br>
    pq.put((0, source))<br>
    while pq.empty() == False:<br>
        u = pq.get()[1]<br>
        print(u, end=" ") # the path having lowest cost<br>
        if u == target:<br>
            break<br>
        for v, c in graph[u]:<br>
            if visited[v] == False:<br>
                visited[v] = True<br>
                pq.put((c, v))<br>
    print()<br>
    
      # for adding edges to graph<br>
def addedge(x, y, cost):<br>
    graph[x].append((y, cost))<br>
    graph[y].append((x, cost))<br>
    
v = int(input("Enter the number of nodes: "))<br>
graph = [[] for i in range(v)] # undirected Graph<br><br>
e = int(input("Enter the number of edges: "))<br>
print("Enter the edges along with their weights:")<br>
for i in range(e):<br>
    x, y, z = list(map(int, input().split()))<br>
    addedge(x, y, z)<br>
source = int(input("Enter the Source Node: "))<br>
target = int(input("Enter the Target/Destination Node: "))<br>
print("\nPath: ", end = "")<br>
best_first_search(source, target, v)<br>

OUTPUT:<br>
Enter the number of nodes: 4<br>
Enter the number of edges: 5<br>
Enter the edges along with their weights:<br>
0 1 1<br>
0 2 1<br>
0 3 2<br>
2 3 2<br>
1 3 3<br>
Enter the Source Node: 2<br>
Enter the Target/Destination Node: 1<br>

Path: 2 0 1 <br>
****************************************************************************************************************
5.Write a Program to Implement Water-Jug Problem using Python.<br>
    #Write a Program to Implement Water-Jug Problem using Python<br>
from collections import defaultdict<br>
jug1, jug2, aim = 4, 3, 2<br>
visited = defaultdict(lambda: False)<br>
def waterJugSolver(amt1, amt2):<br>
    if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):<br>
        print(amt1, amt2)<br>
        return True<br>
    if visited[(amt1, amt2)] == False:<br>
        print(amt1, amt2)<br>
        visited[(amt1, amt2)] = True<br>
        return (waterJugSolver(0, amt2) or<br>
        waterJugSolver(amt1, 0) or<br>
        waterJugSolver(jug1, amt2) or<br>
        waterJugSolver(amt1, jug2) or<br>
        waterJugSolver(amt1 + min(amt2, (jug1-amt1)),amt2 - min(amt2, (jug1-amt1))) or<br>
        waterJugSolver(amt1 - min(amt1, (jug2-amt2)),amt2 + min(amt1, (jug2-amt2))))<br>
    else:<br>
        return False<br>
print("Steps: ")<br>
waterJugSolver(0, 0)<br>

OUTPUT:<br>

Steps: <br>
0 0<br>
4 0<br>
4 3<br>
0 3<br>
3 0<br>
3 3<br>
4 2<br>
0 2<br>
True<br>
****************************************************************************************************
6.Write a Program to Implement Tower of Hanoi using Python.<br>
#Write a Program to Implement Tower of Hanoi using Python.<br>
def TowerOfHanoi(n , source, destination, auxiliary):<br>
    if n==1:<br>
        print ("Move disk 1 from source",source,"to destination",destination)<br>
        return<br>
    TowerOfHanoi(n-1, source, auxiliary, destination)<br>
    print ("Move disk",n,"from source",source,"to destination",destination)<br>
    TowerOfHanoi(n-1, auxiliary, destination, source)<br>
    
n = 3<br>
TowerOfHanoi(n,'X','Y','Z')<br>

OUTPUT:<br>

Move disk 1 from source X to destination Y<br>
Move disk 2 from source X to destination Z<br>
Move disk 1 from source Y to destination Z<br>
Move disk 3 from source X to destination Y<br>
Move disk 1 from source Z to destination X<br>
Move disk 2 from source Z to destination Y<br>
Move disk 1 from source X to destination Y<br>
*******************************************************************************************************************************
