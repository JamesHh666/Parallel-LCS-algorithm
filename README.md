# Python setup:
- python version: 3.7.4
- used module: 
 ``` 
 $ pip install -r requirements.txt
 ```

## CUDA:
cuda verision I use is cuda-10.0 for running on gpu

## See a demo:
- Run without gpu
 ```
 $ python demo.py
 ```

- Run with gpu
 ```
 $ python demo.py --with_cuda
 ```
 
# Run separately
- Run on single cpu with one process:
 ```
 $ python ee5907_lcs.py --mode single_cpu --loop 2 --data_length 1000
 ```

- Run on CPU with multiple processes:
 ```
 $ python ee5907_lcs.py --mode multi_cpus --num_process 2 --loop 2 --data_length 1000
 ```

- Run on GPU:
 ```
 $ python ee5907_lcs.py --mode gpu --loop 2 --data_length 1000
 ```

