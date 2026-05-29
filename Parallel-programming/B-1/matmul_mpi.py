from mpi4py import MPI
import numpy as np
import time

def matmul_serial(A, B):
    """串行三重循环矩阵乘法（用于结果验证）"""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return np.array(C, dtype=np.float32)

def matmul_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 10                     # 矩阵维度，与串行版一致
    np.random.seed(42)         # 固定随机种子

    # ============================================================
    # 注释1: 只在 rank 0 上生成随机矩阵 A 和 B
    # 数据初始位置: rank 0 的内存中
    # ============================================================
    if rank == 0:
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
    else:
        A = None
        B = None

    # 通信原语 1: 广播 B 矩阵
    # 调用: comm.bcast(B, root=0)
    # 数据流向: 从 root=0 进程发送到所有其他进程
    # 作用: 使每个 MPI 进程都拥有完整的 B 矩阵，用于本地计算
    B = comm.bcast(B, root=0)

    # 计算每个进程负责的行块范围 (负载均衡)
    rows_per_proc = N // size
    remainder = N % size
    if rank < remainder:
        start_row = rank * (rows_per_proc + 1)
        local_rows = rows_per_proc + 1
    else:
        start_row = rank * rows_per_proc + remainder
        local_rows = rows_per_proc

    # 准备 Scatterv 所需的 send_counts 和 displs (只在 rank 0 上计算)
    if rank == 0:
        send_counts = []
        displs = []
        for p in range(size):
            if p < remainder:
                r = rows_per_proc + 1
            else:
                r = rows_per_proc
            send_counts.append(r * N)          # 元素个数
            if p == 0:
                displs.append(0)
            else:
                displs.append(displs[-1] + send_counts[p-1])
    else:
        send_counts = None
        displs = None

    # 本地接收缓冲区，用于存放从 rank 0 接收到的行块
    local_A = np.empty((local_rows, N), dtype=np.float32)

    # 通信原语 2: Scatterv (向量散播)
    # 调用: comm.Scatterv([发送缓冲区, send_counts, displs, MPI.FLOAT], 接收缓冲区, root=0)
    # 数据流向: 从 root=0 进程发送到所有进程，每个进程得到不同的行块
    # 作用: 将矩阵 A 按行分块，分发到各个进程，实现数据并行
    comm.Scatterv([A, send_counts, displs, MPI.FLOAT], local_A, root=0)

    # 本地计算乘积: local_C = local_A × B
    start_time = time.time()
    local_C = np.dot(local_A, B)
    local_time = time.time() - start_time

    # 准备 Gatherv 所需的 recv_counts 和 displs (与 Scatterv 相同)
    if rank == 0:
        recv_counts = send_counts
        # 最终结果矩阵 C 在 rank 0 上分配空间
        C = np.empty((N, N), dtype=np.float32)
    else:
        recv_counts = None
        C = None

    # 通信原语 3: Gatherv (向量收集)
    # 调用: comm.Gatherv(发送缓冲区, [接收缓冲区, recv_counts, displs, MPI.FLOAT], root=0)
    # 数据流向: 所有进程将自己的 local_C 发送到 root=0 进程
    # 作用: 将各个进程计算出的结果块收集到 rank 0，组装成完整的 C 矩阵
    comm.Gatherv(local_C, [C, recv_counts, displs, MPI.FLOAT], root=0)

    # 收集所有进程的最大计算时间 (用于性能对比)
    total_time = comm.reduce(local_time, op=MPI.MAX, root=0)

    # 最终在 rank 0 上输出结果并与串行版对比
    if rank == 0:
        print(f"MPI parallel: n={N}, processes={size}, time={total_time:.2f}s")
        print("\nResult C = A × B (MPI Parallel):")
        print(C)

        serial_C = matmul_serial(A, B)
        print("\nResult C = A × B (Serial):")      # 新增
        print(serial_C)
        if np.allclose(C, serial_C, rtol=1e-5):
            print("\n✅ Parallel result matches serial result.")
        else:
            print("\n❌ Mismatch between parallel and serial results.")

if __name__ == "__main__":
    matmul_mpi()
