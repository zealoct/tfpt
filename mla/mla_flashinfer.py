import torch
import flashinfer
import time
import torch.nn as nn

def test_mla_prefill():
    num_qo_heads = 128
    num_kv_heads = 128
    head_dim = 128
    # allocate 128MB workspace buffer
    workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    batch_size = 1
    nnz_kv = 512
    nnz_qo = 512
    qo_indptr = torch.tensor(
        [0, nnz_qo], dtype=torch.int32, device="cuda:0"
    )

    kv_indptr = qo_indptr.clone()
    q = torch.randn(nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    k = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")
    v = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")
    # create auxiliary data structures for batch prefill attention

    prefill_wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=True,
    )
    # compute batch prefill attention, reuse auxiliary data structures

    torch.cuda.synchronize()
    start=time.time()
    loop=1000
    for _ in range(loop):
        _ = prefill_wrapper.run(q, k, v)
    torch.cuda.synchronize()
    end=time.time()

    print(f"time_ave= {(1000000*(end-start)/loop):.3f} us")


def test_mla_decode(num_local_heads = 128,
                    batch_size = 2,
                    head_dim_ckv = 512,
                    head_dim_kpe = 64,
                    page_size = 32):

    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
        backend="fa2"
    )
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() # for decode, each query length is 1
    kv_lens = torch.full((batch_size,), 999, dtype=torch.int32).to(0)
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * 999
    kv_indices = torch.arange(0, batch_size * 999).to(0).int()
    q_nope = torch.randn(batch_size * 1, num_local_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.zeros(batch_size * 1, num_local_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda")
    ckv = torch.randn(batch_size * 999, 1, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
    kpe = torch.zeros(batch_size * 999, 1, head_dim_kpe, dtype=torch.bfloat16, device="cuda")
    sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    mla_wrapper.plan(q_indptr, kv_indptr, kv_indices, kv_lens,
        num_local_heads, head_dim_ckv, head_dim_kpe, page_size,
        False,  # causal
        sm_scale, q_nope.dtype, ckv.dtype,
    )
    torch.cuda.synchronize()
    start=time.time()
    loop=1000
    for _ in range(loop):
        o = mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    torch.cuda.synchronize()
    end=time.time()

    print(f"time_ave= {(1000000*(end-start)/loop):.3f} us")


def test_linear(batch_size, in_features, out_features, device='cuda', num_warmup=10, num_iter=1000):
    # 初始化模型和输入
    linear = nn.Linear(in_features, out_features, dtype=torch.float16).to(device).eval()
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

    # 预热 (避免冷启动误差)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = linear(x)
    torch.cuda.synchronize() if device == 'cuda' else None

    # 正式测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iter):
            _ = linear(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = (time.time() - start_time) * 1000000 / num_iter  # 平均毫秒

    print(f"nn.Linear({in_features}, {out_features}) Batch: {batch_size}\n"
          f"  Latency: {elapsed:.3f} us | ")


def test_bmm(batch_size, m, n, p, device='cuda', num_warmup=10, num_iter=1000):
    """测试 torch.bmm 矩阵乘法性能

    参数:
        batch_size: 批次大小
        m: 左矩阵行数
        n: 左矩阵列数/右矩阵行数
        p: 右矩阵列数
        device: 计算设备 ('cuda' 或 'cpu')
        num_warmup: 预热迭代次数
        num_iter: 正式测试迭代次数
    """
    # 初始化输入张量 (符合 bmm 的 (b,m,n) * (b,n,p) 形状要求)
    a = torch.randn(batch_size, m, n, dtype=torch.float16, device=device)
    b = torch.randn(batch_size, n, p, dtype=torch.float16, device=device)

    # 预热
    for _ in range(num_warmup):
        _ = torch.bmm(a, b)
    torch.cuda.synchronize() if device == 'cuda' else None

    # 正式测试
    start_time = time.time()
    for _ in range(num_iter):
        _ = torch.bmm(a, b)
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = (time.time() - start_time) * 1000000 / num_iter  # 平均毫秒

    print(f"torch.bmm({m}x{n}, {n}x{p}) Batch: {batch_size}\n"
          f"  Latency: {elapsed:.3f} us | ")


print("Prefill")
test_linear(512, 7168, 2112)  # qkv_down
test_linear(512, 1536, 128*192) # q_up
test_linear(512, 512, 128*256) # kv_up
print("prefill_mla")
test_mla_prefill()
test_linear(512, 128*128, 7168) # o_proj

print("")
for bs in [1,2,4,8]:
    print(f"Decode b={bs}")
    test_linear(bs, 7168, 2112) # qkv_down
    test_linear(bs, 1536, 128*192) # q_up
    test_bmm(128, bs, 128, 512) # q_wuk
    print(f"decode mla")
    test_mla_decode(batch_size = bs)
    test_bmm(128, bs, 512, 128) # attn_o_wuv
    test_linear(bs, 128*128, 7168) # o_proj
    print("")
