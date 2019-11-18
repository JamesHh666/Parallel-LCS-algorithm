from numba import cuda
import numpy as np

@cuda.jit
def get_Si(row: int, P_ai: np.ndarray, S: np.ndarray):
    j = cuda.grid(1)

    while j < S.shape[1]:
        t = P_ai[j] != 0
        s = (0 - S[row - 1, j] + t * S[row - 1, P_ai[j] - 1]) != 0
        S[row, j] = S[row - 1, j] + t * (s ^ 1)
        j += cuda.blockDim.x * cuda.gridDim.x