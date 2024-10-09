import torch
import gemm_unified

# fix random seed
torch.manual_seed(0)
def test(kernel_index, dtype=torch.half, a_format='T', b_format='T'):
    a = torch.randn(256, 256, dtype=dtype).cuda()
    b = torch.randn(256, 256, dtype=dtype).cuda()
    if a_format == 'N':
        c_correct = a.T @ b
    elif b_format == 'N':
        c_correct = a @ b.T
    else:
        c_correct = a @ b
    c = gemm_unified.ops.fine_gemm(a, b, kernel_index)
    if not torch.allclose(c, c_correct, rtol=1e-1, atol=1e-1):
        print(f"!!! Incorrect result at kernel {kernel_index}!!!")
        # Compute the absolute difference
        diff = torch.abs(c - c_correct)
        rtol = 1e-5
        atol = 1e-5
        # Identify locations where the difference exceeds the tolerances
        not_close = diff > (atol + rtol * torch.abs(c_correct))

        # Get the indices of these locations
        not_close_indices = torch.nonzero(not_close, as_tuple=True)
        print(not_close_indices)
        breakpoint()

# test(0)
# test(1)
# test(2)
# test(3)
# test(4)
# test(5)
# test(6)
# test(7)
# test(8)
# test(12, a_format='N')
test(13, b_format='N')
test(14, b_format='N')
test(15, b_format='N')
test(16, a_format='N')