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
7.Write a Program to Implement 8-Puzzle Problem using Python.<br>
import copy<br>
from heapq import heappush, heappop<br>
n = 3<br>
row = [ 1, 0, -1, 0 ]<br>
col = [ 0, -1, 0, 1 ]<br>

class priorityQueue:<br>
    def __init__(self):<br>
        self.heap = []<br>
    def push(self, k):<br>
         heappush(self.heap, k)<br>
    def pop(self):<br>
        return heappop(self.heap)<br>
    def empty(self):<br>
        if not self.heap:<br>
            return True<br>
        else:<br>
            return False<br>

class node:<br>
        def __init__(self, parent, mat, empty_tile_pos,cost, level):<br>
            self.parent = parent<br>
            self.mat = mat<br>
            self.empty_tile_pos = empty_tile_pos<br>
            self.cost = cost<br>
            self.level = level<br>
            
        def __lt__(self, nxt):<br>
            return self.cost < nxt.cost<br>
def calculateCost(mat, final) -> int:<br>
    count = 0<br>
    for i in range(n):<br>
        for j in range(n):<br>
            if ((mat[i][j]) and (mat[i][j] != final[i][j])):<br>
                count += 1<br>
    return count<br>

def newNode(mat, empty_tile_pos, new_empty_tile_pos,level, parent, final) -> node:<br>
    new_mat = copy.deepcopy(mat)<br>
    x1 = empty_tile_pos[0]<br>
    y1 = empty_tile_pos[1]<br>
    x2 = new_empty_tile_pos[0]<br>
    y2 = new_empty_tile_pos[1]<br>
    new_mat[x1][y1], new_mat[x2][y2] = new_mat[x2][y2], new_mat[x1][y1]<br>
    cost = calculateCost(new_mat, final)<br>
    new_node = node(parent, new_mat, new_empty_tile_pos,cost, level)<br>
    return new_node<br>

def printMatrix(mat):<br>
    for i in range(n):<br>
        for j in range(n):<br>
            print("%d " % (mat[i][j]), end = " ")<br>
        print()<br>

def isSafe(x, y):<br>
    return x >= 0 and x < n and y >= 0 and y < n<br>

def printPath(root):<br>
    if root == None:<br>
        return<br>
    printPath(root.parent)<br>
    printMatrix(root.mat)<br>
    print()<br>
    
def solve(initial, empty_tile_pos, final):<br>
    pq = priorityQueue()<br>
    cost = calculateCost(initial, final)<br>
    root = node(None, initial,empty_tile_pos, cost, 0)<br>
    pq.push(root)<br>
    while not pq.empty():<br>
        minimum = pq.pop()<br>
        if minimum.cost == 0:<br>
            printPath(minimum)<br>
            return<br>
        for i in range(n):<br>
            new_tile_pos = [minimum.empty_tile_pos[0] + row[i],minimum.empty_tile_pos[1] + col[i], ]<br>
            if isSafe(new_tile_pos[0], new_tile_pos[1]):<br>
                child=newNode(minimum.mat,minimum.empty_tile_pos,new_tile_pos,minimum.level+1,minimum,final,)<br>
                pq.push(child)<br>
initial = [ [ 1, 2, 3 ],[ 5, 6, 0 ],[ 7, 8, 4 ] ]<br>
final = [ [ 1, 2, 3 ],[ 5, 8, 6 ],[ 0, 7, 4 ] ]<br>
empty_tile_pos = [ 1, 2 ]<br>
solve(initial, empty_tile_pos, final)<br>

OUTPUT:<br>

1  2  3  <br>
5  6  0  <br>
7  8  4  <br>

1  2  3  <br>
5  0  6  <br>
7  8  4  <br>

1  2  3  <br>
5  8  6  <br>
7  0  4  <br>

1  2  3  <br>
5  8  6  <br>
0  7  4  <br>

*********************************************************************************************
8. Write a Program to Implement Travelling Salesman problem using Python. <br>
from sys import maxsize <br>
from itertools import permutations <br>
V = 4 <br>

def travellingSalesmanProblem(graph, s): <br>
    # store all vertex apart from source vertex <br>
 vertex = [] <br>
 for i in range(V): <br>
   if i != s: <br>
    vertex.append(i) <br>

     # store minimum weight Hamiltonian Cycle <br>
    min_path = maxsize <br>
    next_permutation=permutations(vertex) <br>
 for i in next_permutation: <br>

       # store current Path weight(cost) <br>
        current_pathweight = 0 <br>

       # compute current path weight <br>
        k = s <br>
        for j in i: <br>
          current_pathweight += graph[k][j] <br>
          k = j <br>
        current_pathweight += graph[k][s] <br>


      # Update minimum <br>
        min_path = min(min_path, current_pathweight) <br>
 return min_path <br>

      # Driver Code <br>
if __name__ == "__main__": <br>

      # matrix representation of graph <br>
 graph = [[0, 10, 15, 20], [10, 0, 35, 25], <br>
          [15, 35, 0, 30], [20, 25, 30, 0]] <br>
s = 0 <br>
print(travellingSalesmanProblem(graph, s)) <br>

OUTPUT: <br>

80 <br>
***************************************************************************************************************************
9.Write a program to implement the FIND-S Algorithm for finding the most specifichypothesis based on a given set of training data samples. Read the training data from a.CSV file.<br>

import csv<br>
with open('Training_examples.csv', 'w', newline='') as file:<br>
    writer = csv.writer(file)<br>
    writer.writerow(["TIME", "WEATHER", "TEMPERATURE",'COMPANY','HUMIDITY','WIND','GOES'])
    writer.writerow(['Morning','Sunny','Warm','Yes','Mild','Strong','Yes'])<br>
    writer.writerow(['Evening','Rainy','Cold','No','Mild','Normal','No'])<br>
    writer.writerow(['Morning','Sunny','Moderate','Yes','Normal','Normal','Yes'])<br>
    writer.writerow(['Evening','Sunny','Cold','Yes','High','Strong','Yes'])<br>
import csv<br>
import pandas as pd<br>
import numpy as np<br>
 
#to read the data in the csv file<br>
data = pd.read_csv("Training_examples.csv")<br>
print(data,"")<br>
 
#making an array of all the attributes<br>
d = np.array(data)[:,:-1]<br>
print("\n The attributes are:\n ",d)<br>
 
#segragating the target that has positive and negative examples<br>
target = np.array(data)[:,-1]<br>
print("\n The target is: ",target)<br>
 
#training function to implement find-s algorithm<br>
def train(c,t):<br>
    for i, val in enumerate(t):<br>
        if val == "Yes":<br>
            specific_hypothesis = c[i].copy()<br>
            break<br>
             
    for i, val in enumerate(c):<br>
        if t[i] == "Yes":<br>
            for x in range(len(specific_hypothesis)):<br>
                if val[x] != specific_hypothesis[x]:<br>
                    specific_hypothesis[x] = '?'<br>
                else:<br>
                    pass<br>
                 
    return specific_hypothesis<br>
 
#obtaining the final hypothesis<br>
print("\n The final hypothesis is:",train(d,target))<br>


OUTPUT:<br>

 TIME WEATHER TEMPERATURE COMPANY HUMIDITY    WIND GOES<br>
0  Morning   Sunny        Warm     Yes     Mild  Strong  Yes<br>
1  Evening   Rainy        Cold      No     Mild  Normal   No<br>
2  Morning   Sunny    Moderate     Yes   Normal  Normal  Yes<br>
3  Evening   Sunny        Cold     Yes     High  Strong  Yes <br>

 The attributes are:<br>
  [['Morning' 'Sunny' 'Warm' 'Yes' 'Mild' 'Strong']<br>
 ['Evening' 'Rainy' 'Cold' 'No' 'Mild' 'Normal']<br>
 ['Morning' 'Sunny' 'Moderate' 'Yes' 'Normal' 'Normal']<br>
 ['Evening' 'Sunny' 'Cold' 'Yes' 'High' 'Strong']]<br>

 The target is:  ['Yes' 'No' 'Yes' 'Yes']<br>

 The final hypothesis is: ['?' 'Sunny' '?' 'Yes' '?' '?']<br>
***************************************************************************
10.Write a program to implement the Candidate-Elimination algorithm, For a given set oftraining data examples stored in a .CSV file.<br>
import csv<br>
with open("ws.csv")as csv_file:<br>
    #csv_file=csv.reader(f)<br>
    #data=list(csv_file)<br>
    readcsv=csv.reader(csv_file,delimiter=',')<br>
    data=[]<br>
    for row in readcsv:<br>
        data.append(row)<br>
    s=data[1][:-1]<br>
    g=[['?'for i in range(len(s))]for j in range(len(s))]<br>
    for i in data:<br>
        if i[-1]=="Yes":<br>
            for j in range(len(s)):<br>
                if i[j]!=s[j]:<br>
                    s[j]='?'<br>
                    g[j][j]='?'<br>
        elif i[-1]=="No":<br>
            for j in range(len(s)):<br>
                if i[j]!=s[j]:<br>
                      g[j][j]=s[j]<br>
                else:<br>
                    g[j][j]="?"<br>
        print("\n steps of candidate elimination algorithm",data.index(i)+1)<br>
        print(s)<br>
        print(g)<br>
    gh=[]<br>
    for i in g:<br>
        for j in i:<br>
            if j!='?':<br>
                gh.append(i)<br>
                break<br>
    print("\nFinal specific hypothesis:\n",s)<br>
    print("\nFinal general hypothesis:\n",gh)    <br>


OUTPUT:<br>
 steps of candidate elimination algorithm 1<br>
['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']<br>
[['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?',<br> '?', '?', '?', '?', '?']]<br>

 steps of candidate elimination algorithm 2<br>
['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']<br>
[['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?',<br> '?', '?', '?', '?', '?']]<br>

 steps of candidate elimination algorithm 3<br>
['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']<br>
[['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'],<br> ['?', '?', '?', '?', '?', 'Same']]<br>

 steps of candidate elimination algorithm 4<br>
['Sunny', 'Warm', '?', 'Strong', '?', '?']<br>
[['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'],<br> ['?', '?', '?', '?', '?', '?']]<br>

Final specific hypothesis:<br>
 ['Sunny', 'Warm', '?', 'Strong', '?', '?']<br>

Final general hypothesis:<br>
 [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?']]<br> 

**************************************************************************************************************
11. Write a Program to Implement N-Queens Problem using Python. <br>
global N<br>
N = 4<br>
def printSolution(board):<br>
    for i in range(N):<br>
        for j in range(N):<br>
            print (board[i][j], end = " ")<br>
        print()<br>
        
def isSafe(board, row, col):<br>
    for i in range(col):<br>
        if board[row][i] == 1:<br>
            return False<br>
    for i, j in zip(range(row, -1, -1),<br>
        range(col, -1, -1)):<br>
        if board[i][j] == 1:<br>
            return False<br>
    for i, j in zip(range(row, N, 1),<br>
        range(col, -1, -1)):<br>
        if board[i][j] == 1:<br>
            return False<br>
    return True<br>

def solveNQUtil(board, col):<br>
    if col >= N:0<br>
        return True<br>
    for i in range(N):<br>
        if isSafe(board, i, col):<br>
            board[i][col] = 1 <br>
            if solveNQUtil(board, col + 1) == True:<br>
                return True<br>
            board[i][col] = 0<br>
    return False<br>
    
def solveNQ():<br>
    board = [ [0, 0, 0, 0],<br>
              [0, 0, 0, 0],<br>
              [0, 0, 0, 0],<br>
              [0, 0, 0, 0] ]<br>
    if solveNQUtil(board, 0) == False:<br>
        print ("Solution does not exist")<br>
        return False<br>
    printSolution(board)<br>
    return True<br>

solveNQ()<br>

OUTPUT:<br>

0 0 1 0 <br>
1 0 0 0 <br>
0 0 0 1 <br>
0 1 0 0 <br>
True<br>

***************************************************************************************************************
12. Write a Program to Implement A* algorithm using Python.<br>
def aStarAlgo(start_node, stop_node):<br>
         
        open_set = set(start_node) <br>
        closed_set = set()<br>
        g = {} #store distance from starting node<br>
        parents = {}# parents contains an adjacency map of all nodes<br>
 
        #ditance of starting node from itself is zero<br>
        g[start_node] = 0<br>
        #start_node is root node i.e it has no parent nodes<br>
        #so start_node is set to its own parent node<br>
        parents[start_node] = start_node<br>
         
         
        while len(open_set) > 0:<br>
            n = None<br>
 
            #node with lowest f() is found<br>
            for v in open_set:<br>
                if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):<br>
                    n = v<br>
             
                     
            if n == stop_node or Graph_nodes[n] == None:<br>
                pass<br>
            else:<br>
                for (m, weight) in get_neighbors(n):<br>
                    #nodes 'm' not in first and last set are added to first<br>
                    #n is set its parent<br>
                    if m not in open_set and m not in closed_set:<br>
                        open_set.add(m)<br>
                        parents[m] = n<br>
                        g[m] = g[n] + weight<br>
                         
     
                    #for each node m,compare its distance from start i.e g(m) to the<br>
                    #from start through n node<br>
                    else:<br>
                        if g[m] > g[n] + weight:<br>
                            #update g(m)<br>
                            g[m] = g[n] + weight<br>
                            #change parent of m to n<br>
                            parents[m] = n<br>
                             
                            #if m in closed set,remove and add to open<br>
                            if m in closed_set:<br>
                                closed_set.remove(m)<br>
                                open_set.add(m)<br>
 
            if n == None:<br>
                print('Path does not exist!')<br>
                return None<br>
 
            # if the current node is the stop_node<br>
            # then we begin reconstructin the path from it to the start_node<br>
            if n == stop_node:<br>
                path = []<br>
 
                while parents[n] != n:<br>
                    path.append(n)<br>
                    n = parents[n]<br>
 
                path.append(start_node)<br>
 
                path.reverse()<br>
 
                print('Path found: {}'.format(path))<br>
                return path<br>
 
 
            # remove n from the open_list, and add it to closed_list<br>
            # because all of his neighbors were inspected<br>
            open_set.remove(n)<br>
            closed_set.add(n)<br>
 
        print('Path does not exist!')<br>
        return None<br>
         
#define fuction to return neighbor and its distance<br>
#from the passed node<br>
def get_neighbors(v):<br>
    if v in Graph_nodes:<br>
        return Graph_nodes[v]<br>
    else:<br>
        return None<br>
#for simplicity we ll consider heuristic distances given<br>
#and this function returns heuristic distance for all nodes<br>
def heuristic(n):<br>
        H_dist = {<br>
            'A': 11,<br>
            'B': 6,<br>
            'C': 99,<br>
            'D': 1,<br>
            'E': 7,<br>
            'G': 0,<br>
             
        }<br>
 
        return H_dist[n]<br>
 
#Describe your graph here  <br>
Graph_nodes = {<br>
    'A': [('B', 2), ('E', 3)],<br>
    'B': [('C', 1),('G', 9)],<br>
    'C': None,<br>
    'E': [('D', 6)],<br>
    'D': [('G', 1)],<br>
     
}
aStarAlgo('A', 'G')

OUTPUT:<br>

Path found: ['A', 'E', 'D', 'G']<br>
['A', 'E', 'D', 'G']<br>
*******************************************************************************************************************************************
11.Write a Program to Implement AO* algorithm using Python.<br>
class Graph:<br>
    def __init__(self, graph, heuristicNodeList, startNode):<br>
        self.graph = graph<br>
        self.H=heuristicNodeList<br>
        self.start=startNode<br>
        self.parent={}<br>
        self.status={}<br>
        self.solutionGraph={}<br>
    
    def applyAOStar(self):<br>
        self.aoStar(self.start, False)<br>

    def getNeighbors(self, v):<br>
        return self.graph.get(v,'')<br>
    
    def getStatus(self,v):<br>
        return self.status.get(v,0)<br>
    
    def setStatus(self,v, val):<br>
        self.status[v]=val<br>
        
    def getHeuristicNodeValue(self, n):<br>
        return self.H.get(n,0)<br>
    
    def setHeuristicNodeValue(self, n, value):<br>
        self.H[n]=value<br>
    
    def printSolution(self):<br>
        print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE STARTNODE:",self.start)<br>
        print("------------------------------------------------------------")<br>
        print(self.solutionGraph)<br>
        print("------------------------------------------------------------")<br>
  
    
    def computeMinimumCostChildNodes(self, v):<br>
        minimumCost=0<br>
        costToChildNodeListDict={}<br>
        costToChildNodeListDict[minimumCost]=[]<br>
        flag=True<br>
        for nodeInfoTupleList in self.getNeighbors(v):<br>
            cost=0<br>
            nodeList=[]<br>
            for c, weight in nodeInfoTupleList:<br>
                cost=cost+self.getHeuristicNodeValue(c)+weight<br>
                nodeList.append(c)<br>
            if flag==True:<br>
                minimumCost=cost<br>
                costToChildNodeListDict[minimumCost]=nodeList<br>
                flag=False<br>
            else:<br>
                if minimumCost>cost:<br>
                    minimumCost=cost<br>
                    costToChildNodeListDict[minimumCost]=nodeList<br>
        return minimumCost, costToChildNodeListDict[minimumCost]<br>
     
    def aoStar(self, v, backTracking):<br>
        print("HEURISTIC VALUES :", self.H)<br>
        print("SOLUTION GRAPH :", self.solutionGraph)<br>
        print("PROCESSING NODE :", v)<br>
        print("-----------------------------------------------------------------------------------------")<br>
        if self.getStatus(v) >= 0:<br>
            minimumCost, childNodeList = self.computeMinimumCostChildNodes(v)<br>
            print(minimumCost, childNodeList)<br>
            self.setHeuristicNodeValue(v, minimumCost)<br>
            self.setStatus(v,len(childNodeList))<br>
            solved=True<br>
            for childNode in childNodeList:<br>
                self.parent[childNode]=v<br>
                if self.getStatus(childNode)!=-1:<br>
                    solved=solved & False<br>
          
        if solved==True:<br>
            self.setStatus(v,-1)<br>
            self.solutionGraph[v]=childNodeList<br>
        if v!=self.start:<br>
            self.aoStar(self.parent[v], True)<br>
        if backTracking==False:<br>
            for childNode in childNodeList:<br>
                self.setStatus(childNode,0)<br>
                self.aoStar(childNode, False)<br>
print ("Graph - 1")<br>
h1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}<br>
graph1 = {<br>
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],<br>
    'B': [[('G', 1)], [('H', 1)]],<br>
    'C': [[('J', 1)]],<br>
    'D': [[('E', 1), ('F', 1)]],<br>
    'G': [[('I', 1)]]<br>
}<br>
G1= Graph(graph1, h1, 'A')<br>
G1.applyAOStar()<br>
G1.printSolution()<br>


OUTPUT:<br>

Graph - 1<br>
HEURISTIC VALUES : {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}<br>
SOLUTION GRAPH : {}<br>
PROCESSING NODE : A<br>
-----------------------------------------------------------------------------------------<br>
10 ['B', 'C']<br>
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}<br>
SOLUTION GRAPH : {}<br>
PROCESSING NODE : B<br>
-----------------------------------------------------------------------------------------<br>
6 ['G']<br>
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}<br>
SOLUTION GRAPH : {}<br>
PROCESSING NODE : A<br>
-----------------------------------------------------------------------------------------<br>
10 ['B', 'C']<br>
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}<br>
SOLUTION GRAPH : {}<br>
PROCESSING NODE : G<br>
-----------------------------------------------------------------------------------------<br>
8 ['I']<br>
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}<br>
SOLUTION GRAPH : {}<br>
PROCESSING NODE : B<br>
-----------------------------------------------------------------------------------------<br>
8 ['H']<br>
HEURISTIC VALUES : {'A': 10, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}<br>
SOLUTION GRAPH : {}<br>
PROCESSING NODE : A<br>
-----------------------------------------------------------------------------------------<br>
12 ['B', 'C']<br>
HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}<br>
SOLUTION GRAPH : {}<br>
PROCESSING NODE : I<br>
-----------------------------------------------------------------------------------------<br>
0 []<br>
HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 0, 'J': 1}<br>
SOLUTION GRAPH : {'I': []}<br>
PROCESSING NODE : G<br>
-----------------------------------------------------------------------------------------<br>
1 ['I']<br>
HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}<br>
SOLUTION GRAPH : {'I': [], 'G': ['I']}<br>
PROCESSING NODE : B<br>
-----------------------------------------------------------------------------------------<br>
2 ['G']<br>
HEURISTIC VALUES : {'A': 12, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}<br>
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}<br>
PROCESSING NODE : A<br>
-----------------------------------------------------------------------------------------<br>
6 ['B', 'C']<br>
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}<br>
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}<br>
PROCESSING NODE : C<br>
-----------------------------------------------------------------------------------------<br>
2 ['J']<br>
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}<br>
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}<br>
PROCESSING NODE : A<br>
-----------------------------------------------------------------------------------------<br>
6 ['B', 'C']<br>
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}<br>
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}<br>
PROCESSING NODE : J<br>
-----------------------------------------------------------------------------------------<br>
0 []<br>
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 0}<br>
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G'], 'J': []}<br>
PROCESSING NODE : C<br>
-----------------------------------------------------------------------------------------<br>
1 ['J']<br>
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 1, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 0}<br>
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G'], 'J': [], 'C': ['J']}<br>
PROCESSING NODE : A<br>
-----------------------------------------------------------------------------------------<br>
5 ['B', 'C']<br>
FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE STARTNODE: A<br>
------------------------------------------------------------<br>
{'I': [], 'G': ['I'], 'B': ['G'], 'J': [], 'C': ['J'], 'A': ['B', 'C']}<br>
------------------------------------------------------------<br>
******************************************************************************************************************************
https://github.com/abhijnha07/AI-ML/blob/main/Classifiers_Clusters.ipynb<br>
https://github.com/Swap77nil/AI-ML/blob/main/Histogram%20Of%20KNN.ipynb<br>
****************************************************************************************************************************
