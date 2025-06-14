# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def rollout_simulation(np.ndarray[np.int32_t, ndim=2] grid, int score, int max_depth, dict rollout_params):
    cdef int depth = 0
    cdef int cur_score = score
    cdef np.ndarray[np.int32_t, ndim=2] np_grid = grid.copy()
    cdef int i, j
    cdef list valid_moves
    cdef str move, best_move
    cdef float best_score, score_eval
    cdef int d

    for d in range(max_depth):
        valid_moves = get_valid_moves(np_grid)
        if not valid_moves:
            break
        if d < 5:
            best_move = None
            best_score = -1e9
            for move in valid_moves:
                test_grid, test_score = move_np(np_grid, cur_score, move)
                score_eval = evaluate_state(test_grid, test_score, rollout_params)
                if score_eval > best_score:
                    best_score = score_eval
                    best_move = move
            if best_move is None:
                best_move = valid_moves[np.random.randint(len(valid_moves))]
            np_grid, cur_score = move_np(np_grid, cur_score, best_move)
        else:
            move = valid_moves[np.random.randint(len(valid_moves))]
            np_grid, cur_score = move_np(np_grid, cur_score, move)
        if is_game_over(np_grid):
            break
    return evaluate_state(np_grid, cur_score, rollout_params)

@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_state(np.ndarray[np.int32_t, ndim=2] grid, int score, dict rollout_params):
    cdef int max_tile = np.max(grid)
    cdef int empty_cells = np.sum(grid == 0)
    cdef float monotonicity = calc_monotonicity(grid)
    cdef float smoothness = calc_smoothness(grid)
    cdef int corner_bonus = 500 if is_max_in_corner(grid) else 0
    corner_bonus += corner_weight(grid, rollout_params)
    cdef int snake_bonus = snake_bonus_func(grid, rollout_params)
    cdef int merge_potential = merge_potential_func(grid, rollout_params)
    return (score + max_tile * 100 + empty_cells * 20 + monotonicity * 10 + smoothness * 2 + corner_bonus
            + snake_bonus + merge_potential)

@cython.boundscheck(False)
@cython.wraparound(False)
def get_valid_moves(np.ndarray[np.int32_t, ndim=2] grid):
    cdef list moves = []
    for move in ['up', 'down', 'left', 'right']:
        new_grid, _ = move_np(grid, 0, move)
        if not np.array_equal(new_grid, grid):
            moves.append(move)
    return moves

@cython.boundscheck(False)
@cython.wraparound(False)
def move_np(np.ndarray[np.int32_t, ndim=2] grid, int score, str direction):
    cdef np.ndarray[np.int32_t, ndim=2] g = grid.copy()
    cdef int s = score
    cdef int i, j, skip, merged_value
    if direction == 'left':
        for i in range(4):
            row = [g[i, k] for k in range(4) if g[i, k] != 0]
            merged = []
            skip = 0
            j = 0
            while j < len(row):
                if not skip and j+1 < len(row) and row[j] == row[j+1]:
                    merged_value = row[j]*2
                    merged.append(merged_value)
                    s += merged_value
                    skip = 1
                else:
                    if not skip:
                        merged.append(row[j])
                    skip = 0
                j += 1 if not skip else 2
            while len(merged) < 4:
                merged.append(0)
            for k in range(4):
                g[i, k] = merged[k]
    elif direction == 'right':
        g = np.fliplr(g)
        g, s = move_np(g, s, 'left')
        g = np.fliplr(g)
    elif direction == 'up':
        g = g.T
        g, s = move_np(g, s, 'left')
        g = g.T
    elif direction == 'down':
        g = g.T
        g, s = move_np(g, s, 'right')
        g = g.T
    return g, s

@cython.boundscheck(False)
@cython.wraparound(False)
def is_game_over(np.ndarray[np.int32_t, ndim=2] grid):
    cdef int i, j
    if np.any(grid == 0):
        return False
    for i in range(4):
        for j in range(3):
            if grid[i, j] == grid[i, j+1]:
                return False
    for j in range(4):
        for i in range(3):
            if grid[i, j] == grid[i+1, j]:
                return False
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
def is_max_in_corner(np.ndarray[np.int32_t, ndim=2] grid):
    cdef int max_tile = np.max(grid)
    cdef int[4] corners = [grid[0,0], grid[0,3], grid[3,0], grid[3,3]]
    for i in range(4):
        if corners[i] == max_tile:
            return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
def corner_weight(np.ndarray[np.int32_t, ndim=2] grid, dict rollout_params):
    cdef int max_tile = np.max(grid)
    cdef int corner_weight = rollout_params.get('corner_weight', 600)
    cdef int bonus = 0
    if grid[3,0] == max_tile:
        bonus += corner_weight
    if grid[3,3] == max_tile:
        bonus += corner_weight
    return bonus

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_monotonicity(np.ndarray[np.int32_t, ndim=2] grid):
    cdef float mono = 0
    for i in range(4):
        mono += mono_line(grid[i, :])
    for i in range(4):
        mono += mono_line(grid[:, i])
    return mono

@cython.boundscheck(False)
@cython.wraparound(False)
def mono_line(np.ndarray[np.int32_t, ndim=1] line):
    cdef float inc = 0
    cdef float dec = 0
    cdef int i
    for i in range(3):
        if line[i] > line[i+1]:
            dec += line[i] - line[i+1]
        else:
            inc += line[i+1] - line[i]
    return -min(inc, dec)

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_smoothness(np.ndarray[np.int32_t, ndim=2] grid):
    cdef float smooth = 0
    cdef int i, j, ni, nj
    cdef float v, nv
    for i in range(4):
        for j in range(4):
            if grid[i, j] == 0:
                continue
            v = np.log2(grid[i, j]) if grid[i, j] > 0 else 0
            for dx, dy in [(0,1),(1,0)]:
                ni, nj = i+dx, j+dy
                if 0<=ni<4 and 0<=nj<4 and grid[ni, nj]>0:
                    nv = np.log2(grid[ni, nj])
                    smooth -= abs(v-nv)
    return smooth

@cython.boundscheck(False)
@cython.wraparound(False)
def snake_bonus_func(np.ndarray[np.int32_t, ndim=2] grid, dict rollout_params):
    cdef int snake_weight = rollout_params.get('snake_weight', 200)
    cdef int idx, i, j
    cdef int bonus = 0
    cdef list snake_order = [
        (3,0),(3,1),(3,2),(3,3),
        (2,3),(2,2),(2,1),(2,0),
        (1,0),(1,1),(1,2),(1,3),
        (0,3),(0,2),(0,1),(0,0)
    ]
    for idx, (i, j) in enumerate(snake_order):
        if grid[i, j] > 0:
            bonus += grid[i, j] * (len(snake_order)-idx) * snake_weight // (2**idx)
    return bonus // 1000

@cython.boundscheck(False)
@cython.wraparound(False)
def merge_potential_func(np.ndarray[np.int32_t, ndim=2] grid, dict rollout_params):
    cdef int merge_weight = rollout_params.get('merge_weight', 100)
    cdef float potential = 0
    cdef int i, j
    for i in range(4):
        for j in range(3):
            if grid[i, j] == grid[i, j+1] and grid[i, j] != 0:
                potential += np.log2(grid[i, j])
    for j in range(4):
        for i in range(3):
            if grid[i, j] == grid[i+1, j] and grid[i, j] != 0:
                potential += np.log2(grid[i, j])
    return int(potential * merge_weight) 