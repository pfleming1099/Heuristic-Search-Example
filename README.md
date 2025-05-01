# Heuristic-Search-Example
An example of a hueristic search algorithm I created during my time at CU Boulder.  
The program uses A* search to pathfind from a chosen start point to a chosen finish point in a customizable grid.  

## Info 

**Grid Setup:** valid_states defines walkable states on a custom-shaped grid.

**Adjacent States:** The adjacent_states(state) function returns valid neighboring states with correct step costs (1 for cardinal directions, √2 for diagonal moves).

**Heuristics:** Several heuristics are implemented:  
heuristic_cols: Difference in x-coordinates (columns).  
heuristic_rows: Difference in y-coordinates (rows).  
heuristic_eucl: Euclidean (straight-line) distance.  
heuristic_max: Maximum of the three heuristics above (used in A*).  

**Path Construction:** 

path(previous, s): Reconstructs the shortest path found.  
pathcost(path, step_costs): Computes the total path cost.  

**A Algorithm:** astar_search(start, goal, state_graph, heuristic, return_cost=False, return_nexp=False):  

Uses a priority queue (Frontier_PQ) based on total cost (path cost + heuristic). Returns the path, total cost, and number of expanded nodes.  

**Visualization:** plot_maze(maze, path=None) plots the grid and optionally overlays the found path using matplotlib.

**State Graph:** state_graph precomputes adjacent connections for each valid state for faster lookup during search.

## How to Run

Ensure the required libraries are installed and run with 'python pathfinder.py'.
