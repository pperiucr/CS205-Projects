# Required library imports

import heapq
import math


##############################################
# PuzzleNode structure to store 
# Its state, parent, cost, and heuristic
# So that object collection can be constructed
##############################################
class PuzzleNode:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

##########################################
# Computing the cost of misplaced tile
# By getting Count of misplaced tiles 
# Tiles out of place from the goal state
##########################################
def misplacedTiles(state, goal):
    # Compares a current puzzle state to the goal state and 
    # counts how many tiles are misplaced.
    return sum(1 for i in range(len(state)) if state[i] != 0 and state[i] != goal[i])

##########################################
# Computing Manhattan Distance 
# Getting sum of distances of each tile 
# from its goal position
##########################################
def manhattanDistance(state, goal, size):
    distance = 0
    # Loops through each tile in the current state
    for i, tile in enumerate(state):
        # Skips the blank tile
        if tile == 0:
            continue
        goal_index = goal.index(tile)
        # Calculates the (row, column) position of the tile in the current state.
        x1, y1 = i // size, i % size
        x2, y2 = goal_index // size, goal_index % size
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

############################################
# Expansion function to return all possible 
# moves from current state
############################################
def expand(node, size):
    moves = []
    index = node.state.index(0)
    x, y = index // size, index % size
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            n_index = nx * size + ny
            new_state = node.state[:]
            new_state[index], new_state[n_index] = new_state[n_index], new_state[index]
            moves.append(PuzzleNode(new_state, node, node.cost + 1))
    return moves

##############################################
# General Search Algorithm 
# Based on the  pseudocode provided in the project
# in built QUEUEING-FUNCTION adds expanded children 
# with updated heuristics to the priority queue
# accepts the starting state, goal state, puzzle size
# and an optional heuristic function
##############################################
def general_search(initial_state, goal_state, size, heuristic_fn=None):

    # Defines a helper function to manage how children are added 
    # to the frontier
    def queueing_function(queue, children):
        # Iterates over each child node
        for child in children:
            if heuristic_fn:
                # Special case for misplaced tiles heuristic,takes only two arguments
                if heuristic_fn == misplacedTiles:
                    child.heuristic = heuristic_fn(child.state, goal_state)
                else:
                    child.heuristic = heuristic_fn(child.state, goal_state, size)
            heapq.heappush(queue, child)
        return queue

    # Initializes the frontier as priority queue and create the initial node.
    frontier = []
    initial_node = PuzzleNode(initial_state, cost=0)
    # Calculates and assigns the heuristic value for the root node
    if heuristic_fn:
        if heuristic_fn == misplacedTiles:
            initial_node.heuristic = heuristic_fn(initial_state, goal_state)
        else:
            initial_node.heuristic = heuristic_fn(initial_state, goal_state, size)
    # Adds the initial node to the frontier.
    heapq.heappush(frontier, initial_node)
    explored = set()

    # Main search loop while the frontier is not empty
    while frontier:
        node = heapq.heappop(frontier)
        state_tuple = tuple(node.state)

        if state_tuple in explored:
            continue
        explored.add(state_tuple)

        # Checks if the goal has been reached; returns the solution node
        if node.state == goal_state:
            return node  # Goal reached

        children = expand(node, size)
        # Adds the children to the frontier with appropriate cost
        frontier = queueing_function(frontier, children)

    # If the loop ends without finding the goal, return None indicating failure
    return None  # Failure

##############################################
# print function the traversal path from 
# initial to goal state
# Each board is printed horizontally to make 
# screen capturing easier for report
##############################################
def printSolution(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    for step in reversed(path):
        print_board(step)
    print(f"\nSolution found in {len(path) - 1} steps.")

##############################################
# Helper function to manage the the display
# the matrix horizontally helping in capturing 
# screenshots for the report
##############################################
def print_board(state):
    size = int(math.sqrt(len(state)))
    row = [' '.join(str(tile) if tile != 0 else '_' for tile in state[i * size:(i + 1) * size]) for i in range(size)]
    print(' | '.join(row))

#########################################
# check function for solution possibility
# Check inversion count
# if it comes out as even grid 
#########################################
def isSolvable(state, size):
    inversions = 0
    state = [tile for tile in state if tile != 0]
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if state[i] > state[j]:
                inversions += 1
    if size % 2 == 1:  # odd grid
        return inversions % 2 == 0
    else:  # even grid
        empty_row = (state.index(0) // size) + 1
        return (inversions + empty_row) % 2 == 0


##############################################
##############################################
# Main program execution
##############################################
##############################################

if __name__ == "__main__":
    # ask for puzzele size
    size = 3
# int(input("Enter puzzle size (e.g., 3 for 3x3): "))
    print(" Enter the initial state as a space-separated list as three separate rows space separated\n")
    # Initializes an empty list to store the puzzle state
    initial_state = []
    # terates over each row of the puzzle. 
    # Prompts the user for a space-separated line 
    # converts it into a list of integers.
    for i in range(size):
        row_input = input(f"Input line {i+1}: ")
        row = list(map(int, row_input.strip().split()))
        while len(row) != size:
            print(f"Line {i+1} must contain exactly {size} numbers.")
            row_input = input(f"Input line {i+1}: ")
            row = list(map(int, row_input.strip().split()))
        initial_state.extend(row)
    # Flattens the row into the initial state list.
    goal_state = list(range(1, size * size)) + [0]

    # Checks puzzle solvability
    if not isSolvable(initial_state, size):
        print("\nThe puzzle is not solvable.")
    else:
        # Runs Uniform Cost Search (UCS) with no heuristic
        print("\n--- Uniform Cost Search ---")
        result = general_search(initial_state, goal_state, size)
        if result:
            printSolution(result)
        else:
            print("No solution found.")

        # Runs A* search using the Misplaced Tile heuristic.
        print("\n--- A* with Misplaced Tile Heuristic ---")
        result = general_search(initial_state, goal_state, size, misplacedTiles)
        if result:
            printSolution(result)
        else:
            print("No solution found.")
        
        # Runs A* search using the Manhattan Distance heuristic.
        print("\n--- A* with Manhattan Distance Heuristic ---")
        result = general_search(initial_state, goal_state, size, manhattanDistance)
        if result:
            printSolution(result)
        else:
            print("No solution found.")
