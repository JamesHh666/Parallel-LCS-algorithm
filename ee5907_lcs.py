import time
import argparse
from numba import cuda
import multiprocessing as mp
from tools.common import load_data, init_S, find_LCS, validate_result
from tools.single_cpu import get_S
from tools.multi_cpus import get_Sikj, generate_input_list, find_P
from tools.gpu import get_Si

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--mode", default="single_cpu",
                        help="which mode to find LCS: \n---- single_cpu\n---- multi_cpus\n---- gpu")
    parser.add_argument("--num_process", type=int, default=-1, help="number of processes to use")
    parser.add_argument("--loop", type=int, default=1, help="number of loops to perform")
    parser.add_argument("--data_length", type=int, default=1000, help="length of generated string")
    parser.add_argument("--new_data", action="store_true", default=False,
                        help="to generate a new version of data with specific length")
    args = parser.parse_args()

    print("\nRunning on {} mode.".format(args.mode))

    # Load data
    data_path = "./data"
    seqA, seqB = load_data(data_path, args.data_length, args.new_data)
    n, m = len(seqA), len(seqB)
    time.sleep(2)  # Give user some time to see if data is loaded correctly

    t1 = []
    t2 = []

    mode = args.mode

    # Check
    if mode not in ["single_cpu", "multi_cpus", "gpu"]:
        raise ValueError("mode should be one of the following: single_cpu/multi_cpus/gpu")
    if mode == "multi_cpus" and args.num_process == -1:
        raise ValueError("Please specify the number of processed you wish to call! (min = 1)\n"
                         "\tuse : --num_process xxx")

    for lo in range(args.loop):
        # Initialize S
        S = init_S(mode, n, m)

        print("---" * 10)
        print("Loop : {}".format(lo+1))

        if mode == "single_cpu":
            start = time.time()

            S = get_S(seqA, seqB, S)

            end = time.time()
            t1.append(end - start)

            print("Time for computing S is: %.4f s" % (t1[-1]))

        elif mode == "multi_cpus":
            # Process count
            max_cpu = mp.cpu_count()
            if args.num_process > max_cpu:
                num_process = max_cpu
            else:
                num_process = args.num_process
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
        print("Average time for finding LCS on {} string is {:.4f} s "
              .format(args.data_length, sum(t1) / args.loop))
    else:
        print("Average time for finding LCS on {} string is {:.4f} s"
              .format(args.data_length, (sum(t1) + sum(t2)) / args.loop))
