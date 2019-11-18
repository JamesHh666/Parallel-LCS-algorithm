# This is a demo which shows the result by single/multiple process and gpu performance on
# "ATGC" sequence of length 100, 1000, 2000, 5000.
# The result will be saved as "demo_result.jpg"

import time
from numba import cuda
import multiprocessing as mp
from tools.common import load_data, init_S, find_LCS, validate_result
from tools.single_cpu import get_S
from tools.multi_cpus import get_Sikj, generate_input_list, find_P
from tools.gpu import get_Si
import matplotlib.pyplot as plt
import argparse

def lcs(mode, data_length, loop, new_data=False, num_process=2):
    print("\nRunning on {} mode.".format(mode))

    # Load data
    data_path = "./data"
    seqA, seqB = load_data(data_path, data_length, new_data)
    n, m = len(seqA), len(seqB)
    time.sleep(2)  # Give user some time to see if data is loaded correctly

    t1 = []
    t2 = []

    mode = mode

    # Check
    if mode not in ["single_cpu", "multi_cpus", "gpu"]:
        raise ValueError("mode should be one of the following: single_cpu/multi_cpus/gpu")
    if mode == "multi_cpus" and num_process == -1:
        raise ValueError("Please specify the number of processed you wish to call! (min = 1)\n"
                         "\tuse : --num_process xxx")

    for lo in range(loop):
        # Initialize S
        S = init_S(mode, n, m)

        print("---" * 10)
        print("Loop : {}".format(lo + 1))

        if mode == "single_cpu":
            start = time.time()

            S = get_S(seqA, seqB, S)

            end = time.time()
            t1.append(end - start)

            print("Time for computing S is: %.4f s" % (t1[-1]))

        elif mode == "multi_cpus":
            # Process count
            max_cpu = mp.cpu_count()
            if num_process > max_cpu:
                num_process = max_cpu
            else:
                num_process = num_process
            print("Running with {} processes".format(num_process))

            # Step 1: find P
            P, t = find_P(seqB)
            t1.append(t)

            # Step 2: find S
            input_list = generate_input_list(m, num_process)
            start = time.time()

            pool = mp.Pool(processes=num_process)
            for i in range(1, n):  # 1 to n-1
                print("{:.2f}".format(i / n), end="\r")  # Show process

                ai = seqA[i]  # str

                # Use num_process CPUs to compute the whole line concurrently
                processes = pool.starmap_async(get_Sikj,
                                               [(js, S[i - 1], P[ai][js[0]: js[-1] + 1]) for js in input_list])
                res = [item for sub_list in processes.get() for item in sub_list]
                S[i][1:] = res

            end = time.time()
            t2.append(end - start)
            print("Time for computing S is: {:.2f} s".format(t2[-1]))
            print("Total computing time: {:.2f} s".format(t2[-1] + t1[-1]))

        elif mode == "gpu":
            d_S = cuda.to_device(S)

            # Step 1: find P
            P, t = find_P(seqB, gpu_mode=True)
            t1.append(t)

            # Step 2: find S
            start = time.time()

            for i in range(1, n):  # 1 to n-1
                print("{:.2f}".format(i / n), end="\r")  # Show process

                ai = seqA[i]  # str
                get_Si(i, P[ai], d_S)

            end = time.time()
            t2.append(end - start)
            print("Time for computing S is: {:.2f} s".format(t2[-1]))
            print("Total computing time: {:.2f} s".format(t2[-1] + t1[-1]))

            S = d_S.copy_to_host()

        # Validation
        LCS = find_LCS(seqA, S)
        validate_result(LCS, seqA, seqB)

    print("---" * 10)

    if mode == "single_cpu":
        average_time = sum(t1) / loop
        print("Average time for finding LCS on {} string is {:.4f} s "
              .format(data_length, average_time))
    else:
        average_time = (sum(t1) + sum(t2)) / loop
        print("Average time for finding LCS on {} string is {:.4f} s"
              .format(data_length, average_time))

    return average_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_cuda", default=False, action="store_true",
                        help="whether use gpu to accelerate. (On if you have cuda-10.0)")
    args = parser.parse_args()

    data_lengths = [100, 500, 1000, 2000, 4000]
    average_time = {"single_cpu": [], "multi_cpus": [], "gpu": []}
    for data_length in data_lengths:
        average_time["single_cpu"].append(lcs("single_cpu", data_length, loop=1))
        average_time["multi_cpus"].append(lcs("multi_cpus", data_length, loop=1))
        if args.with_cuda:
            average_time["gpu"].append(lcs("gpu", data_length, loop=1))

    fig, ax = plt.subplots(1, 1)

    line1, = ax.plot(data_lengths, average_time["single_cpu"], "r", label="SmithWaterman algorithm")
    line2, = ax.plot(data_lengths, average_time["multi_cpus"], 'b', label="New algorithm on CPU", )
    plt.plot(data_lengths, average_time["single_cpu"], 'ro')
    plt.plot(data_lengths, average_time["multi_cpus"], 'bs')

    if args.with_cuda:
        line3, = ax.plot(data_lengths, average_time["gpu"], "g", label="New algorithm on GPU")
        plt.plot(data_lengths, average_time["gpu"], 'g^')

    ax.set_xlabel("sequence lengths")
    ax.set_ylabel("time (seconds) ")
    ax.legend()

    fig.savefig("demo_result.png")
    print("Image saved at './demo_result.png'!")

