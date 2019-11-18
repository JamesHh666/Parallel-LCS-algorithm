import multiprocessing as mp
import time
import numpy as np

def _get_Pi(seq: str, letter: str, gpu_mode: bool) -> dict:
    Pi = {letter: []}
    for j in range(len(seq)):
        if j == 0:
            Pi[letter].append(0)
        elif seq[j] == letter:
            Pi[letter].append(j)
        else:
            Pi[letter].append(Pi[letter][-1])
    if gpu_mode:
        if len(seq) < 255:
            Pi[letter] = np.array(Pi[letter], dtype=np.uint8)
        elif len(seq) < 65535:
            Pi[letter] = np.array(Pi[letter], dtype=np.uint16)
        else:
            Pi[letter] = np.array(Pi[letter], dtype=np.uint32)
    return Pi

def get_Sikj(js: list, Si_prev: list, P_aij_k: list):
    res = []

    # For loop
    for j, P_aij in zip(js, P_aij_k):
        t = P_aij != 0
        s = (0 - Si_prev[j] + t * Si_prev[P_aij - 1]) != 0
        # Sij = Si_prev[j] + t * (s ^ 1)
        res.append(Si_prev[j] + t * (s ^ 1))

    return res

def generate_input_list(length_seqB: int, num_cpu: int):
    char_each_cpu = length_seqB // num_cpu  # number of chars to be computed by each cpu
    char_left = length_seqB % num_cpu

    length = [char_each_cpu] * num_cpu
    for idx in range(char_left):
        length[idx] += 1

    res = []
    pos = 0
    for n in range(num_cpu):
        if n == 0:
            res.append([x for x in range(1, length[n])])
            pos += length[n]
        else:
            res.append([x for x in range(pos, pos + length[n])])
            pos += length[n]
    return res

def find_P(seqB: str, gpu_mode=False):
    start = time.time()

    pool = mp.Pool(processes=4)
    processes = pool.starmap_async(_get_Pi, [(seqB, letter, gpu_mode) for letter in "ATCG"])
    Pi_list = processes.get()
    P = {**Pi_list[0], **Pi_list[1], **Pi_list[2], **Pi_list[3]}

    end = time.time()
    t = end - start
    print("Time for computing P is: {:.2f} s".format(t))
    return P, t

