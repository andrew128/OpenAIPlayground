from copy import deepcopy
# --------------------------
ZOMBIE = "z"
CAR = "c"
ICE_CREAM = "i"
EMPTY = "*"

grid = [
    [ICE_CREAM, EMPTY],
    [ZOMBIE, CAR]
]

for row in grid:
    print(' '.join(row))

# ---------------------------
# Environment state that holds current grid and car position
class State:
    def __init__(self, grid, car_pos):
        self.grid = grid
        self.car_pos = car_pos

    def __eq__(self, other):
        return isinstance(other, State) and
                self.grid == other.grid and
                self.car_pos == other.car_pos

    def __hash__(self):
        return hash(str(self.grid) + str(self.car_pos))

    def __str__(self):
        return f"State(grid={self.grid}, car_pos={self.car_pos})"
# -----------------------------
# All possible actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]
# --------------------------------
# Initial state
start_state = State(grid=grid, car_pos=[1, 1])
# --------------------------------
# Transition dynamics
def act(state, action):

    def new_car_pos(state, action):
        p = deepcopy(state.car_pos)
        if action == UP:
            p[0] = max(0, p[0] - 1)
        elif action == DOWN:
            p[0] = min(len(state.grid) - 1, p[0] + 1)
        elif action == LEFT:
            p[1] = max(0, p[1] - 1)
        elif action == RIGHT:
            p[1] = min(len(state.grid[0]) - 1, p[1] + 1)
        else:
            raise ValueError(f"Unknown action {action}")
        return p

    p = new_car_pos(state, action)
    grid_item = state.grid[p[0]][p[1]]

    new_grid = deepcopy(state.grid)

    if grid_item == ZOMBIE:
        reward = -100
        is_done = True
        new_grid[p[0]][p[1]] += CAR
    elif grid_item == ICE_CREAM:
        reward = 1000
        is_done = True
        new_grid[p[0]][p[1]] += CAR
    elif grid_item == EMPTY:
        reward = -1
        is_done = False
        old = state.car_pos
        new_grid[old[0]][old[1]] = EMPTY
        new_grid[p[0]][p[1]] = CAR
    elif grid_item == CAR:
        reward = -1
        is_done = False
    else:
        raise ValueError(f"Unknown grid item {grid_item}")

    return State(grid=new_grid, car_pos=p), reward, is_done
# --------------------------
# Learning to Drive
