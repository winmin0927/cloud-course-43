from mpi4py import MPI
import numpy as np
import argparse
import time

def parallel_matmul(A, B, comm):
    """
    所有进程已持有完整的 A 和 B，直接按行分块计算，最后 Gather 合并。
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    n = A.shape[0]

    # 计算每个进程负责的行区间
    rows_per_proc = n // size
    remainder = n % size
    if rank < remainder:
        start = rank * (rows_per_proc + 1)
        end = start + rows_per_proc + 1
    else:
        start = rank * rows_per_proc + remainder
        end = start + rows_per_proc

    local_A = A[start:end, :]          # 行块视图（零拷贝）
    local_rows = end - start
    local_C = np.zeros((local_rows, n), dtype=np.float64)

    # 三重循环矩阵乘法
    for i in range(local_rows):
        for k in range(n):
            aik = local_A[i, k]
            row_B = B[k, :]
            for j in range(n):
                local_C[i, j] += aik * row_B[j]

    # 收集所有局部结果到根进程
    C_parts = comm.gather(local_C, root=0)

    if rank == 0:
        return np.vstack(C_parts)   # 按行拼接成完整矩阵
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=800, help="矩阵大小")
    parser.add_argument("--repeat", type=int, default=3, help="重复次数")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 主进程生成随机矩阵，然后广播给所有进程
    if rank == 0:
        np.random.seed(42)
        A = np.random.rand(args.N, args.N).astype(np.float64)
        B = np.random.rand(args.N, args.N).astype(np.float64)
        print(f"Matrix size: {args.N}x{args.N}, MPI processes: {size}")
    else:
        A = None
        B = None

    A = comm.bcast(A, root=0)
    B = comm.bcast(B, root=0)
    comm.Barrier()  # 确保所有进程都收到数据

    times = []
    for i in range(args.repeat):
        comm.Barrier()
        start = MPI.Wtime()
        C = parallel_matmul(A, B, comm)
        comm.Barrier()
        end = MPI.Wtime()
        if rank == 0:
            elapsed = end - start
            times.append(elapsed)
            print(f"Run {i+1}: {elapsed:.4f} seconds")

    if rank == 0:
        avg = np.mean(times)
        std = np.std(times)
        print(f"\n--- Results for {size} processes ---")
        print(f"Average time: {avg:.4f} ± {std:.4f} seconds")
        print(f"RESULT: processes={size}, avg={avg:.6f}, std={std:.6f}")

    MPI.Finalize()

if __name__ == "__main__":
    main()