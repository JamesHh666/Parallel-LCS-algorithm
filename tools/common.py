import os
import numpy as np
# from shutil import rmtree
import random

def init_S(mode: str, n: int, m: int):
    if mode == "multi_cpus":
        # Use numpy in multiprocessing will dramatically increase the processing time
        S = [[0 for _ in range(m)] for _ in range(n)]
    else:
        if m < 2 ^ 8:
            S = np.zeros((n, m), dtype=np.uint8)
        elif m < 2 ^ 16:
            S = np.zeros((n, m), dtype=np.uint16)
        else:
            S = np.zeros((n, m), dtype=np.uint32)
    return S

def _isSubsequence(sub_seq: str, t: str) -> bool:
    if sub_seq == "":
        return False
    for i in sub_seq:
      idx = t.find(i)
      if idx == -1:
        return False
      else:
        t = t[idx+1:]
    return True


def find_LCS(seqA, S):
    res = ""

    if isinstance(S, list):
        i = len(S)
        j = len(S[0])


    elif isinstance(S, np.ndarray):
        i, j = S.shape

    i -= 1
    j -= 1
    while True:
        if S[i][j] == 0:
            break
        elif S[i][j] != max(S[i-1][j], S[i][j-1]):
            res = seqA[i] + res
            i -= 1
            j -= 1
        elif S[i][j] == S[i-1][j]:
            i -= 1
        elif S[i][j] == S[i][j-1]:
            j -= 1

    print("The length of LCS is: %d" % len(res))
    print("Some letters of LCS are {}".format(res[:20]))
    return res

def validate_result(LCS, seqA, seqB):
    if _isSubsequence(LCS, seqA):
        print("LCS is the subseq of seqA")
    else:
        print("LCS is not the subseq of seqA")
    if _isSubsequence(LCS, seqB):
        print("LCS is the subseq of seqB")
    else:
        print("LCS is not the subseq of seqB")
    return 0

def _data_generator(length: int, path: str):
    ALPHABET = "AGTC"

    for i in range(2):
        res = ""

        for _ in range(length):
            letter = random.choice(ALPHABET)
            res += letter

        data_name = "{}_{}.txt".format(length, i)
        save_path = os.path.join(path, data_name)

        with open(save_path, "w") as f:
            f.write(res)

        print("Data of length {} generated at {}".format(length, save_path))

    return 0

def load_data(path: str, length: int, new_data: bool) -> tuple:
    if not os.path.exists(path):
        os.mkdir(path)
        _data_generator(length, path)
    elif new_data:
        _data_generator(length, path)
    else:
        # Check whether data of length xx exist
        exist = False
        files = os.listdir(path)
        for file in files:
            if file.split("_")[0] == str(length):
                exist = True
                break
        if not exist:
            _data_generator(length, path)

    files = os.listdir(path)
    SR = ["", ""] # Only load 2 strings because the algorithm can only find LCS between 2 strings

    cnt = 0
    for file in files:
        if file.split("_")[0] == str(length):
            with open(os.path.join(path, file), 'r') as f:
                for line in f:
                    SR[cnt] += line.replace('\n','').strip()
            cnt += 1

    seqA = "0" + SR[0]
    seqB = "0" + SR[1]

    print("Successfully load data with length {} from {}".format(len(seqA) - 1, path))

    return seqA, seqB