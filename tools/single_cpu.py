def get_S(seqA, seqB, S):
    for i in range(1, len(seqA)):  # 1 to n-1
        print("{:.2f}".format(i / len(seqA)), end="\r")  # Show process

        for j in range(1, len(seqB)):
            if seqA[i] == seqB[j]:
                S[i, j] = S[i-1, j-1] + 1
            else:
                S[i, j] = max(S[i-1, j], S[i, j-1])

    return S