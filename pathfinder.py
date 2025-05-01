import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import defaultdict

# added packages
import heapq
from matplotlib import colors

#map here
valid_states = [(1 ,16),(2 ,16),(3 ,16),(4 ,16),(5 ,16),(6 ,16),(7 ,16),(8 ,16),(9 ,16),(10,16),(11,16),(12,16),(13,16),(14,16),(15,16),(16,16),(17,16),(18,16),(19,16),(20,16),
                (1 ,15),(2 ,15),(3 ,15),(4 ,15),(5 ,15),(6 ,15),(7 ,15),(8 ,15),(9 ,15),(10,15),(11,15),(12,15),(13,15),(14,15),(15,15),(16,15),(17,15),(18,15),(19,15),(20,15),
                        (2 ,14),(3 ,14),(4 ,14),(5 ,14),(6 ,14),(7 ,14),(8 ,14),(9 ,14),(10,14),                                (15,14),(16,14),(17,14),(18,14),(19,14),(20,14),
                                (3 ,13),(4 ,13),(5 ,13),(6 ,13),(7 ,13),(8 ,13),(9 ,13),(10,13),                                (15,13),(16,13),(17,13),(18,13),(19,13),(20,13),
                                        (4 ,12),(5 ,12),(6 ,12),(7 ,12),(8 ,12),(9 ,12),                                        (15,12),(16,12),(17,12),(18,12),(19,12),(20,12),
                                                (5 ,11),(6 ,11),(7 ,11),(8 ,11),(9 ,11),                                        (15,11),(16,11),(17,11),(18,11),(19,11),(20,11),
                                                (5 ,10),(6 ,10),(7 ,10),(8 ,10),(9 ,10),                                (14,10),(15,10),(16,10),(17,10),(18,10),(19,10),(20,10),(21,10),(22,10),(23,10),(24,10),(25,10),
                                                (5 ,9 ),(6 ,9 ),(7 ,9 ),(8 ,9 ),(9 ,9 ),                                (14,9 ),(15,9 ),(16,9 ),(17,9 ),(18,9 ),(19,9 ),(20,9 ),(21,9 ),(22,9 ),(23,9 ),(24,9 ),(25,9 ),
                                                (5 ,8 ),(6 ,8 ),(7 ,8 ),(8 ,8 ),(9 ,8 ),(10,8 ),(11,8 ),(12,8 ),(13,8 ),(14,8 ),(15,8 ),(16,8 ),(17,8 ),(18,8 ),(19,8 ),(20,8 ),(21,8 ),(22,8 ),(23,8 ),(24,8 ),(25,8 ),
                                                (5 ,7 ),(6 ,7 ),(7 ,7 ),(8 ,7 ),(9 ,7 ),(10,7 ),(11,7 ),(12,7 ),(13,7 ),(14,7 ),(15,7 ),(16,7 ),(17,7 ),(18,7 ),(19,7 ),(20,7 ),(21,7 ),(22,7 ),(23,7 ),(24,7 ),(25,7 ),
                                                (5 ,6 ),(6 ,6 ),(7 ,6 ),(8 ,6 ),(9 ,6 ),(10,6 ),(11,6 ),(12,6 ),(13,6 ),(14,6 ),(15,6 ),(16,6 ),(17,6 ),(18,6 ),(19,6 ),(20,6 ),(21,6 ),(22,6 ),(23,6 ),(24,6 ),(25,6 ),
                                                (5 ,5 ),(6 ,5 ),(7 ,5 ),(8 ,5 ),(9 ,5 ),(10,5 ),(11,5 ),(12,5 ),(13,5 ),(14,5 ),(15,5 ),(16,5 ),(17,5 ),(18,5 ),(19,5 ),(20,5 ),(21,5 ),(22,5 ),(23,5 ),(24,5 ),(25,5 ),
                                        (4 ,4 ),(5 ,4 ),(6 ,4 ),(7 ,4 ),(8 ,4 ),(9 ,4 ),(10,4 ),(11,4 ),(12,4 ),(13,4 ),(14,4 ),(15,4 ),(16,4 ),(17,4 ),(18,4 ),(19,4 ),(20,4 ),(21,4 ),(22,4 ),(23,4 ),(24,4 ),(25,4 ),
                                (3 , 3),(4 ,3 ),(5 ,3 ),(6 ,3 ),(7 ,3 ),(8 ,3 ),(9 ,3 ),(10,3 ),(11,3 ),(12,3 ),(13,3 ),(14,3 ),(15,3 ),(16,3 ),(17,3 ),(18,3 ),(19,3 ),(20,3 ),(21,3 ),(22,3 ),(23,3 ),(24,3 ),(25,3 ),
                        (2, 2 ),(3 , 2),(4 ,2 ),(5 ,2 ),(6 ,2 ),(7 ,2 ),(8 ,2 ),(9 ,2 ),(10,2 ),(11,2 ),(12,2 ),(13,2 ),(14,2 ),(15,2 ),(16,2 ),(17,2 ),(18,2 ),(19,2 ),(20,2 ),(21,2 ),(22,2 ),(23,2 ),(24,2 ),(25,2 ),
                (1 ,1 ),(2, 1 ),(3 , 1),(4 ,1 ),                                                                                                                                                        (24,1 ),(25,1 )]

def adjacent_states(state):
    x, y = state
    adjacent = {}
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    
    for dx, dy in directions:
        newState = (x + dx, y + dy)
        if newState in valid_states:
            #cardinal directions
            if dx == 0 or dy == 0:
                stepCost = 1
            #diagonal
            else:
                stepCost = np.sqrt(2)
            adjacent[newState] = stepCost
    return adjacent

print(adjacent_states((1,15)))

def heuristic_cols(state, goal):
    xstate = state[0]
    xgoal = goal[0]
    return abs(xstate - xgoal)

def heuristic_rows(state, goal):
    ystate = state[1]
    ygoal = state[1]
    return abs(ystate - ygoal)

def heuristic_eucl(state, goal):
    xstate = state[0]
    ystate = state[1]
    
    xgoal = goal[0]
    ygoal = state[1]
    return np.sqrt((xstate - xgoal)**2 + (ystate - ygoal)**2)

def heuristic_max(state, goal):
    return max(heuristic_cols(state, goal), heuristic_rows(state, goal), heuristic_eucl(state, goal))

def path(previous, s): 
    '''
    `previous` is a dictionary chaining together the predecessor state that led to each state
    `s` will be None for the initial state
    otherwise, start from the last state `s` and recursively trace `previous` back to the initial state,
    constructing a list of states visited as we go
    '''
    if s is None:
        return []
    else:
        return path(previous, previous[s])+[s]

def pathcost(path, step_costs):
    '''
    add up the step costs along a path, which is assumed to be a list output from the `path` function above
    '''
    cost = 0
    for s in range(len(path)-1):
        cost += step_costs[path[s]][path[s+1]]
    return cost

class Frontier_PQ:
    ''' frontier class for uniform search, ordered by path cost '''
    
    def __init__(self, start, cost):
        self.states = {}
        self.q = []
        self.add(start, cost)
        
    def add(self, state, cost):
        ''' push the new state and cost to get there onto the heap'''
        heapq.heappush(self.q, (cost, state))
        self.states[state] = cost

    def pop(self):
        (cost, state) = heapq.heappop(self.q)  # get cost of getting to explored state
        self.states.pop(state)    # and remove from frontier
        return (cost, state)

    def replace(self, state, cost):
        ''' found a cheaper route to `state`, replacing old cost with new `cost` '''
        self.states[state] = cost
        for i, (oldcost, oldstate) in enumerate(self.q):
            if oldstate==state and oldcost > cost:
                self.q[i] = (cost, state)
                heapq._siftdown(self.q, 0, i) # now i is posisbly out of order; restore
        return

def astar_search(start, goal, state_graph, heuristic, return_cost=False, return_nexp=False):
    '''A* search from `start` to `goal`
    start = initial state
    goal = goal state
    heuristic = function for estimated cost to goal (function name)
    return_cost = logical (True/False) for whether or not to return the total path cost
    return_nexp = logical (True/False) for whether or not to return the number of nodes expanded
    '''         
    frontier = Frontier_PQ(start, 0)
    previous = {start : None}
    explored = {}
    n_exp = 0
    while frontier:
        s = frontier.pop()
        n_exp += 1
        #print('exploring {}'.format(s[1]))
        if s[1] == goal:
            if return_cost:
                if return_nexp:
                    return (path(previous, s[1]), pathcost(path(previous, s[1]), state_graph), n_exp)
                    #return (path(previous, s[1]), pathcost(path(previous, s[1]), state_graph), len(explored))
                else:
                    return (path(previous, s[1]), pathcost(path(previous, s[1]), state_graph))
            else:
                if return_nexp:
                    return (path(previous, s[1]), n_exp)
                else:
                    return path(previous, s[1])
        explored[s[1]] = pathcost(path(previous, s[1]), state_graph)
        for s2 in state_graph[s[1]]:
            newcost = explored[s[1]] + state_graph[s[1]][s2] + heuristic(s2, goal)
            if (s2 not in explored) and (s2 not in frontier.states):
                frontier.add(s2, newcost)
                previous[s2] = s[1]
            elif (s2 in frontier.states) and (frontier.states[s2] > newcost):
                frontier.replace(s2, newcost)
                previous[s2] = s[1]
                
start = (1,15)
goal = (25,9)
state_graph = {state: adjacent_states(state) for state in valid_states}
result = astar_search(start, goal, state_graph, heuristic_max, True, True)
print("Shortest path:", result[0])
print("Total path cost:", result[1])
print("Number of nodes expanded:", result[2])

def plot_maze(maze, path=None):
    ''' visualize the binary `maze` (assumed numpy array) and solution `path` (if provided)'''

    nrow, ncol = maze.shape
    #print(nrow, ncol)
    # create colormap
    cmap = colors.ListedColormap(['coral', 'slategray'])

    fig, ax = plt.subplots()
    ax.imshow(maze, cmap=cmap, origin='lower')
    
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k')
    ax.set_xticks(np.arange(-.5, ncol, 1))
    ax.set_yticks(np.arange(-.5, nrow, 1))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    
    # now draw the solution path, if one is provided
    # 
    if path:
        path_x = [point[0] - 1 for point in path]  # Adjust x coordinates
        path_y = [point[1] - 1 for point in path]  # Adjust y coordinates
        ax.plot(path_x, path_y, c='black')
    
    plt.show()
#     if path == maze_sol_bfs:
#         ax.set_title("Breadth First Search Path")
    
#     elif path == maze_sol_dfs:
#         ax.set_title("Depth First Search Path")
#     plt.show()

#convert coordinate grid to 1s and 0s
maze = np.zeros((16, 25))
for state, adjacents in state_graph.items():
    for adjacent in adjacents:
        x, y = state
        maze[y-1, x-1] = 1 #if obstabcle, place 1

plot_maze(maze, path=result[0])