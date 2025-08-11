import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# load the grid file into an array
grid_str = open('grid.txt', 'r').read()
index = grid_str.find("\n")
t = 3


def make_grid():
    grid = np.array([['BLANK' for _ in range(150)] for _ in range(150)])
    open_spaces = []
    for y in range(150):
        for x in range(150):
            grid[y][x] = grid_str[2*y*150 + 2*x]
            if grid[y][x] == '.':
                open_spaces.append((x, y))
    return grid, open_spaces


def get_adjacent(x, y):
    ret_list = []
    for x_offset in range(-1, 2):
        for y_offset in range(-1, 2):
            compare_x = x + x_offset
            compare_y = y + y_offset
            if compare_y < 0 or compare_y > 149 or compare_x < 0 or compare_x > 149 or (compare_x == x and compare_y == y):
                continue
            ret_list.append((compare_x, compare_y))
    return ret_list


def is_satisfied(x: int, y: int, current_type: str, grid: list):
    adjacent = 0
    if current_type == '.':
        return True
    adjacent_list = get_adjacent(x, y)
    for adj_x, adj_y in adjacent_list:
        if grid[adj_y][adj_x] == current_type:
            adjacent += 1
    return adjacent >= t


def update_satisfies_lists(grid, open_spaces):
    sum_adj_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    x_array = (grid == 'X')
    o_array = (grid == 'O')

    x_neighbor_count = convolve2d(x_array, sum_adj_kernel, mode='same', boundary='wrap')
    o_neighbor_count = convolve2d(o_array, sum_adj_kernel, mode='same', boundary='wrap')

    satisfies_x = []
    satisfies_o = []
    for y in range(len(x_neighbor_count)):
        for x in range(len(x_neighbor_count[0])):
            if grid[y][x] == '.':
                if x_neighbor_count[y][x] >= t:
                    satisfies_x.append((x, y))
                if o_neighbor_count[y][x] >= t:
                    satisfies_o.append((x, y))
    return satisfies_x, satisfies_o


def move_agent(grid, open_spaces, start_x: int, start_y: int):
    agent_to_move = grid[start_y][start_x]
    satisfies_x, satisfies_o = update_satisfies_lists(grid, open_spaces)

    if agent_to_move == '.':
        print('PROBLEM: cant move an empty space')

    end_x = 200
    end_y = 200
    if len(satisfies_x) == 0 and len(satisfies_o) == 0:
        # print('No satisfying cells for either type')
        return False
    if agent_to_move == 'X':
        if len(satisfies_x) == 0:
            return True
        end_x, end_y = satisfies_x[random.randint(0, len(satisfies_x) - 1)]
    elif agent_to_move == 'O':
        if len(satisfies_o) == 0:
            return True
        end_x, end_y = satisfies_o[random.randint(0, len(satisfies_o) - 1)]
    else:
        print('PROBLEM: attempted to move an element that is not X, O, or .')

    if grid[end_y][end_x] != '.':
        print('PROBLEM: Attempted to move into an occupied space', grid[end_y][end_x])
        return True

    grid[end_y][end_x] = agent_to_move
    grid[start_y][start_x] = '.'

    # The cell that the agent was now moved to is no longer empty
    open_spaces.remove((end_x, end_y))
    open_spaces.append((start_x, start_y))
    return True


# def calculate_fraction(grid):
#     sum_adj_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
#     x_array = (grid == 'X')
#     o_array = (grid == 'O')
#     agents_array = (grid != '.')
#     x_neighbor_count = convolve2d(x_array, sum_adj_kernel, mode='same', boundary='wrap')
#     o_neighbor_count = convolve2d(o_array, sum_adj_kernel, mode='same', boundary='wrap')
#     all_neighbor_count = convolve2d(agents_array, sum_adj_kernel, mode='same', boundary='wrap')
#     ret_val = (np.sum(x_neighbor_count/all_neighbor_count)/10000) + (np.sum(o_neighbor_count/all_neighbor_count)/10000)
#     print(ret_val)
#     return ret_val


def run_loop(grid, num_rounds, open_spaces, satisfies_x, satisfies_o):
    same_type_fractions = []
    for i in range(num_rounds):
        same_type_neighbors = 0
        total_neighbors = 0
        satisfied_agents = 0

        for y in range(len(grid)):
            for x in range(len(grid[0])):
                # # Calculate fraction of same-type neighbors
                if grid[y][x] != '.':
                    neighbors = get_adjacent(x, y)
                    for neighbor_x, neighbor_y in neighbors:
                        if grid[neighbor_y][neighbor_x] != '.' and grid[neighbor_y][neighbor_x] == grid[y][x]:
                            same_type_neighbors += 1
                        if grid[neighbor_y][neighbor_x] != '.':
                            total_neighbors += 1
                # Attempt to move unsatisfied agents
                if not is_satisfied(x, y, grid[y][x], grid):
                    # If there are no cells that satisfy either type of agent, return the list we already have
                    if not move_agent(grid, open_spaces, x, y):
                        if total_neighbors > 0:
                            same_type_fractions.append(same_type_neighbors / total_neighbors)
                        else:
                            same_type_fractions.append(0)
                        # same_type_fractions.append(calculate_fraction(grid))
                        return same_type_fractions
                else:
                    satisfied_agents += 1
        if satisfied_agents == 22500:
            break

        same_type_fractions.append(same_type_neighbors / total_neighbors)
        # same_type_fractions.append(calculate_fraction(grid))
    return same_type_fractions


def run_model():
    same_type_fraction_list = np.zeros(1000)
    num_rounds = 1000
    trial_count = 20
    initial_grid, initial_open_spaces = make_grid()

    for i in range(trial_count):
        grid = initial_grid.copy()
        open_spaces = initial_open_spaces.copy()
        for x, y in open_spaces:
            if grid[y][x] != '.':
                print('PROBLEM: element in open spaces does not have a dot on the grid')

        satisfies_x, satisfies_o = update_satisfies_lists(grid, open_spaces)

        for x, y in satisfies_x:
            if (x, y) not in open_spaces:
                print('PROBLEM: element in satisfies x is not open')
        for x, y in satisfies_o:
            if (x, y) not in open_spaces:
                print('PROBLEM: element in satisfies o is not open')

        returned_value = np.array(run_loop(grid, num_rounds, open_spaces, satisfies_x, satisfies_o))
        returned_value.resize(1000)
        # for plotting purposes we will say that after the model terminated(due to deadlock or all agents satisfied)
        # all values remained the same for the remaining rounds, so we will copy the last nonzero value into all the
        # trailing indices with values of zero
        for j in range(len(returned_value)):
            if returned_value[j] == 0:
                returned_value[j] = returned_value[j - 1]

        same_type_fraction_list = same_type_fraction_list + returned_value
        # print(same_type_fraction_list)
    same_type_fraction_list = same_type_fraction_list / trial_count
    round_number_array = np.arange(1, 1001)

    # print('T =', t)
    # print(same_type_fraction_list)

    plt.plot(round_number_array, same_type_fraction_list)


for t_val in range(2, 7):
    t = t_val
    run_model()
plt.title('Plot of Average Fraction of Same-Type Neighbors vs. Round Number')
plt.ylabel('Average Fraction of Same-Type Neighbors')
plt.xlabel('Round Number')
plt.show()
