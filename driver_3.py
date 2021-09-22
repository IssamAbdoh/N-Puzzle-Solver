# -*- coding: utf-8 -*-
"""
A python project for solving n-puzzle in any size(should be a square)

Finished on Wed Sep 22 22:11:49 2021

@author: Essam Abdo

python version 3.7.9
"""

from collections import deque
import time
import psutil
import argparse
from heapq import heappush, heappop, heapify
from math import sqrt
import sys

"""
deque is imported for lifo and fifo operation for stack(dfs) and queue(bfs)
time is imported so we can find the accurate time
psutil is imported for measuring RAM usage
argparse is imported in order to use to program from a command prompt
heapq is imported for using priority_queue(ast)
math is imported for using the sqrt function
sys is imported in order to get the maximum number possible in python
"""


def printLine():
    print(9*"/|\\")
    """
    for printing a line to orgnize output
    """


def returnLine():
    return (9*"/|\\")+"\n"
    """
    for returning a line to orgnize output
    created for file writing
    """


class state():
    def __init__(self, array, parent):
        self.board_size = int(sqrt(len(array)))  # Row size of the puzzle
        self._arr = array
        self.constructNPuzzle()
        self.parent = parent
        self.move = None
        self.depth = 0
        if self.parent:
            self.depth = parent.depth + 1

        self.estimated_cost = self.depth + self.__h2()  # f = cost + h

    def constructNPuzzle(self):  # constructing a board structure for the puzzle
        if len(self._arr) == self.board_size:
            return
        arr2 = [i for i in self._arr]
        self._arr = []
        h = []
        for j in arr2:
            h.append(j)
            if len(h) == self.board_size:
                self._arr.append(h)
                h = []

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, value):
        self._arr = value
        self.constructNPuzzle()  # adjustment after input

    def printBoard(self):
        for row in self._arr:
            print(row)
        print()

    def returnBoard(self):
        s = ""
        for row in self.arr:
            s += row.__repr__()+"\n"
        return s

    def __eq__(self, other):
        """
        this method was created in order to make the objects
        hashable so we can store them in a set
        """
        h1 = [item for row in self.arr for item in row]
        h2 = [item for row in other.arr for item in row]
        for i in range(self.board_size * self.board_size):
            if h1[i] != h2[i]:
                return False
        return True

    def children(self):
        """
        this method returns a list containing all the possible changes
        for the current state
        """
        n = []
        zi = 0
        zj = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.arr[i][j] == 0:
                    zi = i
                    zj = j
                    break
        i = zi
        j = zj
        # "UDLR" order; that is, [‘Up’, ‘Down’, ‘Left’, ‘Right’]
        if 0 <= i-1:  # up
            self.arr[i][j], self.arr[i-1][j] = self.arr[i -
                                                        1][j], self.arr[i][j]  # do the move
            x = state([item for row in self.arr for item in row],
                      self)  # create a copy
            x.move = "Up"  # set the movement that leads for this copy
            n.append(x)  # add the copy to the list
            # undo the move so we can continue finding the other children
            self.arr[i][j], self.arr[i-1][j] = self.arr[i-1][j], self.arr[i][j]
        if self.board_size > i+1:  # down
            self.arr[i][j], self.arr[i+1][j] = self.arr[i+1][j], self.arr[i][j]
            x = state([item for row in self.arr for item in row], self)
            x.move = "Down"
            n.append(x)
            self.arr[i][j], self.arr[i+1][j] = self.arr[i+1][j], self.arr[i][j]
        if 0 <= j-1:  # left
            self.arr[i][j], self.arr[i][j-1] = self.arr[i][j-1], self.arr[i][j]
            x = state([item for row in self.arr for item in row], self)
            x.move = "Left"
            n.append(x)
            self.arr[i][j], self.arr[i][j-1] = self.arr[i][j-1], self.arr[i][j]
        if self.board_size > j+1:  # right
            self.arr[i][j], self.arr[i][j+1] = self.arr[i][j+1], self.arr[i][j]
            x = state([item for row in self.arr for item in row], self)
            x.move = "Right"
            n.append(x)
            self.arr[i][j], self.arr[i][j+1] = self.arr[i][j+1], self.arr[i][j]

        return n

    def __h1(self):  # _misplace_tiles
        """
        this method is never used in the program
        it was coded just in case I needed it for calculating the 
        estimated cost , not never used
        I felt that I should not negate the effort spent coding it :D
        so I kept it here :p
        """
        current = [item for row in self.arr for item in row]
        # here I am counting 0 as a tile
        goal = [i for i in range(self.board_size * self.board_size)]
        h1 = sum([int(current[i] != goal[i])
                 for i in range(self.board_size * self.board_size)])
        return h1

    def __h2(self):  # _manhattan_distance
        """
        used for calulcating the estimated cost for reaching a the goal state from our current state
        """
        h2 = 0

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.arr[i][j] == 0:
                    continue
                h2 += (abs(i-(self.arr[i][j]//self.board_size)) +
                       abs(j-(self.arr[i][j] % self.board_size)))

        return h2

    def __hash__(self):
        """
        this method was created in order to make the objects
        hashable so we can store them in a set
        """
        return hash(self.returnBoard())

    def __repr__(self):
        return self.returnBoard()

    def __lt__(self, other):
        """
        this method was created in order to make the objects
        sortable for the heapq
        """
        return self.estimated_cost < other.estimated_cost

    def __gt__(self, other):
        """
        this method was created in order to make the objects
        sortable for the heapq
        """
        return self.estimated_cost > other.estimated_cost


class solve():
    def __init__(self):
        """
            path_to_goal: the sequence of moves taken to reach the goal
            cost_of_path: the number of moves taken to reach the goal
            nodes_expanded: the number of nodes that have been expanded
            fringe_size: the size of the frontier set when the goal node is found
            max_fringe_size: the maximum size of the frontier set in the lifetime of the algorithm
            search_depth: the depth within the search tree when the goal node is found
            max_search_depth: the maximum depth of the search tree in the lifetime of the algorithm
            running_time: the total running time of the search instance, reported in seconds
            max_ram_usage: the maximum RAM usage in the lifetime of the process as measured by the
            ru_maxrss attribute in the resource module, reported in megabytes
        """
        self.path_to_goal = []
        self.cost_of_path = 0
        self.nodes_expanded = 0
        # how many nodes did I see their children ,
        # which is the number of explored noder - 1(the one that I stopped searching after (the goal))
        self.fringe_size = 0
        self.max_fringe_size = 0
        self.search_depth = 0
        self.max_search_depth = 0
        self.running_time = 0
        self.max_ram_usage = 0

    def __success(self, initial_st, finalState, expanded_nodes, size_of_fringe, mx_size_of_fringe, mx_search_depth, time_needed, mx_ram, search_type):
        # calculating (constructing the path to goal)
        h = finalState
        while h.move != None:
            self.path_to_goal.append(h.move)
            h = h.parent

        #assigning and preparing
        self.path_to_goal.reverse()
        self.cost_of_path = len(self.path_to_goal)
        self.nodes_expanded = expanded_nodes
        self.fringe_size = size_of_fringe
        self.max_fringe_size = mx_size_of_fringe
        self.search_depth = self.cost_of_path
        self.max_search_depth = mx_search_depth
        self.running_time = round(time_needed, 8)
        self.max_ram_usage = round(
            mx_ram / (1024.0 ** 2), 8)  # in megabytes MB

        # print to the console
        printLine()
        print(search_type)
        print("initial State :")
        initial_st.printBoard()
        print("goal State :")
        finalState.printBoard()
        print("path_to_goal", self.path_to_goal)
        print("cost_of_path", self.cost_of_path)
        print("nodes_expanded", self.nodes_expanded)
        print("fringe_size", self.fringe_size)
        print("max_fringe_size", self.max_fringe_size)
        print("search_depth", self.search_depth)
        print("max_search_depth", self.max_search_depth)
        print("running_time", self.running_time)
        print("max_ram_usage", self.max_ram_usage)
        printLine()

        # print to the file
        self.output(search_type, initial_st, finalState)

    def __reset_all_variables(self):
        self.path_to_goal = []
        self.cost_of_path = 0
        self.nodes_expanded = 0
        self.fringe_size = 0
        self.max_fringe_size = 0
        self.search_depth = 0
        self.max_search_depth = 0
        self.running_time = 0
        self.max_ram_usage = 0

    def bfs(self, initialSt, goalSt):  # Breadth­First Search
        """
        sets were used in order to shortening the lookup searching time
        so the total time can be at least as possible
        """
        self.__reset_all_variables()

        start = time.perf_counter()

        frontier = deque()  # deque will be treated as a queue
        frontier.append(initialSt)
        explored = set()
        frontier_U_explored = set()  # for fasten up the lookup time

        max_frontier_size = 0
        max_ram_used = psutil.virtual_memory().used

        while len(frontier) != 0:
            currentState = frontier.popleft()
            explored.add(currentState)
            frontier_U_explored.add(currentState)

            if goalSt == currentState:
                end = time.perf_counter()
                self.__success(initialSt,
                               currentState,
                               len(explored)-1,
                               len(frontier),
                               max_frontier_size,
                               frontier[-1].depth,
                               end-start,
                               max_ram_used,
                               "bfs")
                return True

            for child in currentState.children():
                if child not in frontier_U_explored:
                    frontier.append(child)

            max_frontier_size = len(frontier) if len(
                frontier) > max_frontier_size else max_frontier_size
            max_ram_used = psutil.virtual_memory().used if psutil.virtual_memory(
            ).used > max_ram_used else max_ram_used
        return False

    def dfs(self, initialSt, goalSt):  # Depth­First Search
        """
        sets were used in order to shortening the lookup searching time
        so the total time can be at least as possible
        """

        self.__reset_all_variables()

        start = time.perf_counter()

        frontier = deque()  # deque will be treated as a stack
        frontier.append(initialSt)
        frontier_U_explored = set()
        frontier_U_explored.add(initialSt)  # for fasten up the lookup time
        explored = set()

        max_frontier_size = 0
        max_ram_used = psutil.virtual_memory().used
        max_depth = initialSt.depth

        while len(frontier):
            currentState = frontier.pop()
            explored.add(currentState)
            frontier_U_explored.add(currentState)

            max_depth = currentState.depth if currentState.depth > max_depth else max_depth

            if goalSt == currentState:

                end = time.perf_counter()

                self.__success(initialSt,
                               currentState,
                               len(explored)-1,
                               len(frontier),
                               max_frontier_size,
                               max_depth,
                               end-start,
                               max_ram_used,
                               "dfs")
                return True

            h = currentState.children()
            h.reverse()
            for child in h:
                if child not in frontier_U_explored:
                    frontier.append(child)
                    frontier_U_explored.add(child)

            max_frontier_size = len(frontier) if len(
                frontier) > max_frontier_size else max_frontier_size
            max_ram_used = psutil.virtual_memory().used if psutil.virtual_memory(
            ).used > max_ram_used else max_ram_used

        return False

    def ast(self, initialSt, goalSt, is_ida=False, f_limit=sys.maxsize):  # A­Star Search
        """
        this algorithm will be used for the IDA too
        so it was made complicated in order to maintain efficiecy for 
        both algorithms (ast) and (ida)
        is_ida : is this call for ida ? if yes certain operations will
        be done
        f_limit : the limit of deepening of this iteration (this field was
        added for ida algorithm)
        """

        """
        sets were used in order to shortening the lookup searching time
        so the total time can be at least as possible
        """
        self.__reset_all_variables()

        start = time.perf_counter()

        frontier = []
        frontier_set = set()
        heapify(frontier)
        """
        this algorithm relays on the estimated cost
        so we used the heapq(priority queue) for sorting our nodes
        in an efficient way (ascending)
        """
        heappush(frontier, initialSt)
        frontier_set.add(initialSt)
        explored = set()
        frontier_U_explored = set()  # set was used in order to fasten up the lookup time
        frontier_U_explored.add(initialSt)

        max_frontier_size = 0
        max_ram_used = psutil.virtual_memory().used
        max_depth = initialSt.depth

        while len(frontier):
            current_state = heappop(frontier)
            frontier_set.remove(current_state)
            explored.add(current_state)
            frontier_U_explored.add(current_state)

            max_depth = current_state.depth if current_state.depth > max_depth else max_depth

            if goalSt == current_state:
                end = time.perf_counter()

                if not is_ida:
                    self.__success(initialSt,
                                   current_state,
                                   len(explored)-1,
                                   len(frontier),
                                   max_frontier_size,
                                   max_depth,
                                   end-start,
                                   max_ram_used,
                                   "ast")
                    return True
                else:
                    """FOR IDA
                    if we reached a success , this will be for sure the
                    optimal solution for the n-puzzle so we will be 
                    sending it to ida to continuing what is needed to 
                    be done .
                    """
                    return current_state, len(explored)-1, len(frontier), max_frontier_size, max_depth, max_ram_used

            for child in current_state.children():
                if child not in frontier_U_explored:
                    if child.estimated_cost < f_limit:
                        heappush(frontier, child)
                        frontier_U_explored.add(child)
                        frontier_set.add(child)
                elif child in frontier_set:
                    """ 
                    reset child weight if duplicate found in frontier 
                    """
                    frontier.remove(child)
                    heappush(frontier, child)
                    frontier_set.remove(child)
                    frontier_set.add(child)

            max_frontier_size = len(frontier) if len(
                frontier) > max_frontier_size else max_frontier_size
            max_ram_used = psutil.virtual_memory().used if psutil.virtual_memory(
            ).used > max_ram_used else max_ram_used

        return False

    def ida(self, initialSt, goalSt):  # IDA­Star Search
        """
        we will keep doing ast algorithm 
        but everytime with an increased limit
        so we can take a wider look into the tree
        until we find a success !
        """

        self.__reset_all_variables()

        start = time.perf_counter()

        f_limit = initialSt.estimated_cost
        while True:
            # performing ast
            result = self.ast(initialSt, goalSt, True, f_limit)

            if result:  # if ast with the specified limit succeeded
                finalState = result[0]
                expanded_nodes = result[1]
                size_of_fringe = result[2]
                mx_size_of_fringe = result[3]
                mx_search_depth = result[4]
                mx_ram = result[5]

                end = time.perf_counter()

                # return current_state,len(explored)-1,len(frontier),max_frontier_size,max_depth,max_ram_used
                self.__success(initialSt,
                               finalState,
                               expanded_nodes,
                               size_of_fringe,
                               mx_size_of_fringe,
                               mx_search_depth,
                               end-start,
                               mx_ram,
                               "ida")
                return True
            else:
                """
                if ast with the specified limit did not succeed , 
                then increase the limit and retry again
                """
                f_limit += int(sqrt(f_limit))

        return False

    def output(self, type_of_search, initial_st, goal_st):
        s = returnLine()
        s += type_of_search+"\n\n"
        s += "initial State :\n"+initial_st.returnBoard() + "\n"
        s += "goal State :\n"+goal_st.returnBoard() + "\n"
        s += "path_to_goal : " + str(self.path_to_goal) + "\n"
        s += "cost_of_path : " + str(self.cost_of_path) + "\n"
        s += "nodes_expanded : " + str(self.nodes_expanded) + "\n"
        s += "fringe_size : " + str(self.fringe_size) + "\n"
        s += "max_fringe_size : " + str(self.max_fringe_size) + "\n"
        s += "search_depth : " + str(self.search_depth) + "\n"
        s += "max_search_depth : " + str(self.max_search_depth) + "\n"
        s += "running_time : " + str(self.running_time) + "\n"
        s += "max_ram_usage : " + str(self.max_ram_usage) + "\n"
        s += returnLine()

        with open("output.txt", "a") as f:
            f.write(s)
        print("Print to output.txt has been done successfully !")


def main():
    """#for the normal running of the program
    s = state([1, 2, 5, 3, 4, 0, 6, 7, 8], None)
    g = state([0, 1, 2, 3, 4, 5, 6, 7, 8], None)
    #g = state([1,0,2,3,4,5,6,7,8])
    solver = solve()
    print(solver.bfs(s, g))
    #n = state([7,2,4,5,0,6,8,3,1],None)

    s = state([15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0], None)

    g = state([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], None)
    g2 = state([1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], None)
    #g = state([1,0,2,3,4,5,6,7,8])
    solver = solve()
    #print(solver.bfs(g2, g))
    n = state([7, 2, 4, 5, 0, 6, 8, 3, 1], None)

    #print(solver.ast(n, g))
    #print(solver.ida(n, g))
    """
    
    #///////////////////////////////////////////////////////////
    
    #"""#for the commad prompt usage
    parser = argparse.ArgumentParser("Week 2 Project: Search Algorithms")
    parser.add_argument("method", help="Searching Algorithm Name", type=str)
    parser.add_argument("board", help="Initial State of the board")

    args = parser.parse_args()
    solver = solve()
    try:
        initial_state = [int(i) for i in args.board.split(",")]
        s = state(initial_state, None)
    except:
        print("An error with input has been occured .")
        exit()

    g = state([i for i in range(len(initial_state))], None)

    if args.method == "bfs":
        print(solver.bfs(s, g))
    elif args.method == "dfs":
        print(solver.dfs(s, g))
    elif args.method == "ast":
        print(solver.ast(s, g))
    else:
        print(solver.ida(s, g))

    #"""

    return

if __name__ == "__main__":
    main()
