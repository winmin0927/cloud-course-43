from mpi4py import MPI
import numpy as np
import argparse
import time

def parallel_matmul_nb(A, B, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    n = A.shape[0]

    # 计算当前进程负责的行区间
    rows_per_proc = n // size
    remainder = n % size
    if rank < remainder:
        start = rank * (rows_per_proc + 1)
        end = start + rows_per_proc + 1
    else:
        start = rank * rows_per_proc + remainder
        end = start + rows_per_proc

    local_A = A[start:end, :]          
    local_rows = end - start
    local_C = np.zeros((local_rows, n), dtype=np.float64)

    # ---------------------------------------------------------
    # 核心优化：将任务切分为多个块（Chunks），以实现计算与通信的重叠
    # ---------------------------------------------------------
    num_chunks = 4  # 分成4块
    chunk_size = (local_rows + num_chunks - 1) // num_chunks
    
    reqs = []

    # 0号进程：提前发起所有的非阻塞接收 (Irecv)
    if rank == 0:
        recv_buffers = []
        for p in range(1, size):
            p_start = p * (rows_per_proc + 1) if p < remainder else p * rows_per_proc + remainder
            p_end = p_start + (rows_per_proc + 1 if p < remainder else rows_per_proc)
            p_rows = p_end - p_start
            p_chunk_size = (p_rows + num_chunks - 1) // num_chunks

            for c in range(num_chunks):
                c_start = c * p_chunk_size
                c_end = min((c + 1) * p_chunk_size, p_rows)
                if c_start < c_end:
                    buf = np.empty((c_end - c_start, n), dtype=np.float64)
                    req = comm.Irecv(buf, source=p, tag=p * 100 + c)
                    recv_buffers.append((p_start + c_start, p_start + c_end, buf, req))

    # 所有进程：逐块进行计算，算完立刻异步发送
    for c in range(num_chunks):
        c_start = c * chunk_size
        c_end = min((c + 1) * chunk_size, local_rows)
        if c_start >= c_end:
            break

        # 1. 计算当前块 (纯 CPU 操作)
        for i in range(c_start, c_end):
            for k in range(n):
                aik = local_A[i, k]
                row_B = B[k, :]
                for j in range(n):
                    local_C[i, j] += aik * row_B[j]

        # 2. 计算完毕后立即触发非阻塞发送 (Isend)，让网卡在后台工作
        if rank != 0:
            # 使用 .copy() 防止后续计算或内存回收破坏正在发送的缓冲区
            req = comm.Isend(local_C[c_start:c_end, :].copy(), dest=0, tag=rank * 100 + c)
            reqs.append(req)

    # 0号进程：等待所有异步接收完成，并将接收缓冲区的数据拼接成完整矩阵
    if rank == 0:
        final_C = np.zeros((n, n), dtype=np.float64)
        final_C[start:end, :] = local_C  # 填入自己计算的部分

        for p_start_idx, p_end_idx, buf, req in recv_buffers:
            req.Wait()  # 阻塞直到该块数据接收完毕
            final_C[p_start_idx:p_end_idx, :] = buf

        return final_C
    else:
        # 工作进程：必须等待自己的发送任务在后台全部走完才能退出函数
        MPI.Request.Waitall(reqs)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=800, help="矩阵大小")
    parser.add_argument("--repeat", type=int, default=3, help="重复次数")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        np.random.seed(42)
        A = np.random.rand(args.N, args.N).astype(np.float64)
        B = np.random.rand(args.N, args.N).astype(np.float64)
        print(f"Matrix size: {args.N}x{args.N}, MPI processes: {size} (NON-BLOCKING)")
    else:
        A = None
        B = None

    A = comm.bcast(A, root=0)
    B = comm.bcast(B, root=0)
    comm.Barrier()

    times = []
    for i in range(args.repeat):
        comm.Barrier()
        start = MPI.Wtime()
        C = parallel_matmul_nb(A, B, comm)  # 非阻塞版本
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
