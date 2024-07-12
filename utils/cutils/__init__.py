from pathlib import Path
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
from torch.nn import functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import os

path = Path(__file__).parent
build_dir = path / "build"
build_dir.mkdir(exist_ok=True)
sources = [str(p) for p in path.glob("srcs/*.*") if p.suffix in [".cpp", ".cu"]]

cutils = load("cutils_", sources=sources, extra_cflags=["-O3", "-mavx2", "-funroll-loops"], extra_cuda_cflags=["-Xptxas","-v"],
              verbose=True, build_directory=build_dir)

def next_prime(x) -> int:
    r"""
    Finds the next prime, x included.           
    x should be >= 3 for a correct result.
    """
    x = int(x) | 1
    for i in range(x, 2*x, 2):
        prime = True
        for j in range(3, int(i**0.5) + 1, 2):
            if i % j == 0:
                prime = False
                break
        if prime:
            return i

def grid_subsampling(xyz: torch.Tensor, grid_size: float, hash_size: float=1.) -> torch.Tensor:
    r"""
    xyz: N x 3, float, non-negative coordinates
    grid_size: float, positive
    hash_size: How large the hash table should be relative to the original point cloud size.
                If estimated downsampling ratio is k, i.e., ori_size = k * subsampled_size,
                then recommended value is 2~3 / k.
                Must be greater than 1 / real_k
    return value: M, int64, selected indices
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
    if xyz.stride(0) != 3:
        xyz = xyz.contiguous()
    size = xyz.shape[0] * hash_size
    size = next_prime(size)
    table = torch.zeros((size,), dtype=torch.int64)
    storage = torch.empty((size * 3,), dtype=torch.int64)
    indices = cutils.grid_subsampling(xyz, grid_size, table, storage)
    return indices

def grid_subsampling_test(xyz: torch.Tensor, grid_size: float, hash_size: float=1., pick=0) -> torch.Tensor:
    r"""
    xyz: N x 3, float, non-negative coordinates
    grid_size: float, positive
    hash_size: How large the hash table should be relative to the original point cloud size.
                If estimated downsampling ratio is k, i.e., ori_size = k * subsampled_size,
                then recommended value is 2~3 / k.
                Must be greater than 1 / real_k
    pick:  the nth point in the same grid to pick, random picked if actual resident points < pick
    return value: M, int64, selected indices
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
    if xyz.stride(0) != 3:
        xyz = xyz.contiguous()
    size = xyz.shape[0] * hash_size
    size = next_prime(size)
    table = torch.zeros((size,), dtype=torch.int64)
    storage = torch.empty((size * 4,), dtype=torch.int64)
    indices = cutils.grid_subsampling_test(xyz, grid_size, table, storage, pick)
    return indices

class KDTree():
    r"""
    kdt = KDTree(xyz) 
    indices, squared_dists = kdt.knn(query_xyz, k=16, ordered=True)
    indices: int32
    dists: float

    Setting ordered = False (default) can be 1.1-1.2x faster. 
    If there are not enough neighbors, the nearest point is used for padding. 
    Resources (reference to xyz, built tree) are freed when kdt goes out of life scope.
    """
    def __init__(self, xyz: torch.Tensor, max_leaf_size=20):
        assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
        if xyz.stride(0) != 3:
            xyz = xyz.contiguous()
        # reserve xyz for knn search
        self.xyz = xyz
        self.n = xyz.shape[0]
        self.tree, self.pca = cutils.kdtree_build(xyz, max_leaf_size)
    
    def __del__(self):
        cutils.kdtree_free(self.tree, self.pca)
    
    def knn(self, query: torch.Tensor, k=1, ordered=False):
        assert query.ndim == 2 and query.shape[1] == 3 and query.dtype == torch.float
        if query.stride(0) != 3:
            query = query.contiguous()
        queries = query.shape[0]
        nbrs = min(self.n, k)
        if self.n < k : ordered = True
        indices = torch.empty((queries, nbrs), dtype=torch.int32)
        dists = torch.empty((queries, nbrs), dtype=torch.float)
        cutils.kdtree_knn(self.tree, query, indices, dists, ordered)
        if self.n < k:
            indices = torch.cat([indices, indices[:, :1].expand(-1, k - self.n)], dim=1)
            dists = torch.cat([dists, dists[:, :1].expand(-1, k - self.n)], dim=1)
        return indices, dists

class KEMP(Function):
    r"""
    f_i = max{f_j | j in knn_i} - f_i
    output = knn_edge_maxpooling(feature, knn, training=True)  

    Only cuda version supported.

    feature: BNC, float / half
    knn:     BNk, int64
    output:  BNC, float / half

    While not training and gradient is not required, 
    backward indices are not saved. Consumed time and space reduced slightly.
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, feature: torch.Tensor, knn: torch.Tensor, training: bool=True) -> torch.Tensor:
        assert feature.is_cuda and knn.is_cuda
        assert feature.is_contiguous() and knn.is_contiguous() and feature.shape[:2] == knn.shape[:2]
        assert knn.dtype == torch.int64
        if feature.dtype == torch.half:
            assert feature.shape[-1] % 8 == 0, "KEMP half precision impl only supports multiples of 8 as feature dim"
        elif feature.dtype == torch.float32:
            assert feature.shape[-1] % 4 == 0, "KEMP single precision impl only supports multiples of 4 as feature dim"
        else:
            raise NotImplementedError

        output = torch.empty_like(feature)
        if training or feature.requires_grad:
            indices = torch.empty_like(feature, dtype=torch.int32)
            if feature.dtype == torch.half:
                cutils.half_aligned_knn_edge_maxpooling_forward(output, indices, feature, knn)  
            else:
                cutils.aligned_knn_edge_maxpooling_forward(output, indices, feature, knn)
            ctx.save_for_backward(indices)
        else:
            if feature.dtype == torch.half:
                cutils.half_aligned_knn_edge_maxpooling_infer(output, feature, knn)
            else:
                cutils.aligned_knn_edge_maxpooling_infer(output, feature, knn)
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        output = -grad
        indices, = ctx.saved_tensors
        if grad.dtype == torch.half:
            cutils.half_knn_edge_maxpooling_backward(output, indices, grad)  
        else: 
            cutils.knn_edge_maxpooling_backward(output, indices, grad)
        return output, None, None

knn_edge_maxpooling = KEMP.apply


import numpy as np
def rand_rot():
    dim = 1
    axis = torch.randn(dim, 3)
    axis = F.normalize(axis, dim=1)
    angle = torch.rand(dim, 1, 1) * torch.pi
    A = torch.zeros(dim, 3, 3)
    A[:, 0, 1] = -axis[:, 2]
    A[:, 0, 2] = axis[:, 1]
    A[:, 1, 0] = axis[:, 2]
    A[:, 1, 2] = -axis[:, 0]
    A[:, 2, 0] = -axis[:, 1]
    A[:, 2, 1] = axis[:, 0]
    M = A * angle.sin() + torch.bmm(A, A) * (1 - angle.cos()) + torch.eye(3)
    return M.reshape(3, 3)

# gpt wrote this
# works 
def fibonacci_sphere(samples=1,randomize=True):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        radius = np.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * radius
        z = np.sin(phi) * radius

        points.append([x,y,z])

    return points

# gpt taught me this trick
def init_coor(dim, all_dist_ = None):
    if all_dist_ is None:
        all_dist = init_coor.all_dist
    else:
        all_dist = init_coor.all_dist = all_dist_
    s = len(all_dist) // (dim+2)
    length = all_dist[s::s][:dim]
    length = length[torch.randperm(dim)]
    coor = torch.tensor(fibonacci_sphere(dim), dtype=torch.float) @ rand_rot() * length.view(-1, 1)
    return coor

def generate_spse_matrix(dim, all_dist = None):
    coor = init_coor(dim, all_dist)
    scale = torch.empty(dim, 3).uniform_(0, 1) 
    axis = torch.randn(dim, 3)
    axis = F.normalize(axis, dim=1)
    angle = torch.rand(dim, 1) * torch.pi
    M = torch.cat([coor, scale, axis, angle], dim=1)
    return M

def prepare_m(m):
    m = m.view(-1, 10)
    dim = m.shape[0]
    coor = m[:, :3]
    scale = m[:, 3:6]
    axis = m[:, 6:9]
    angle = m[:, 9].view(-1, 1, 1)
    axis = F.normalize(axis, dim=1)
    A = torch.zeros(dim, 3, 3, device=axis.device)
    A[:, 0, 1] = -axis[:, 2]
    A[:, 0, 2] = axis[:, 1]
    A[:, 1, 0] = axis[:, 2]
    A[:, 1, 2] = -axis[:, 0]
    A[:, 2, 0] = -axis[:, 1]
    A[:, 2, 1] = axis[:, 0]
    M = A * angle.sin() + torch.bmm(A, A) * (1 - angle.cos()) + torch.eye(3, device=axis.device)
    M = M @ torch.diag_embed(scale)
    return torch.cat([M.view(-1, 9), coor], dim=1)

class NSPSE4(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, xyz: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor, training: bool=True) -> torch.Tensor:
        B, N, _ = xyz.shape
        C = weight.numel() // 16
        output = torch.empty(B, N, C, device=xyz.device)
        back_idx = torch.empty(B, N, C, dtype=torch.int32, device=xyz.device)
        cutils.knn_spse_4_forward(output, back_idx, xyz, knn, weight)
        if training or weight.requires_grad:
            ctx.save_for_backward(xyz, weight, back_idx)
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        xyz, weight, back_idx = ctx.saved_tensors
        C = grad.shape[2]
        wgrad = torch.zeros(256, 16, C, device=xyz.device)
        cutils.knn_spse_4_backward(wgrad, grad, back_idx, xyz, weight)
        wgrad = wgrad.sum(dim=0).reshape(*weight.shape)
        return None, None, wgrad, None

knn_spse_4 = NSPSE4.apply

class NSPSE4N(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, xyz: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor, training: bool=True) -> torch.Tensor:
        B, N, _ = xyz.shape
        C = weight.numel() // 19
        output = torch.empty(B, N, C, device=xyz.device)
        back_idx = torch.empty(B, N, C, dtype=torch.int32, device=xyz.device)
        cutils.knn_spse_4n_forward(output, back_idx, xyz, knn, weight)
        if training or weight.requires_grad:
            ctx.save_for_backward(xyz, weight, back_idx)
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        xyz, weight, back_idx = ctx.saved_tensors
        C = grad.shape[2]
        wgrad = torch.zeros(256, 19, C, device=xyz.device)
        cutils.knn_spse_4n_backward(wgrad, grad, back_idx, xyz, weight)
        wgrad = wgrad.sum(dim=0).reshape(*weight.shape)
        return None, None, wgrad, None

knn_spse_4n = NSPSE4N.apply

class NSPSE(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, xyz: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor, training: bool=True) -> torch.Tensor:
        B, N, _ = xyz.shape
        C = weight.numel() // 12
        output = torch.empty(B, N, C, device=xyz.device)
        back_idx = torch.empty(B, N, C, dtype=torch.int32, device=xyz.device)
        cutils.knn_spse_forward(output, back_idx, xyz, knn, weight)
        if training or weight.requires_grad:
            ctx.save_for_backward(xyz, weight, back_idx)
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        xyz, weight, back_idx = ctx.saved_tensors
        C = grad.shape[2]
        wgrad = torch.zeros(256, 12, C, device=xyz.device)
        cutils.knn_spse_backward(wgrad, grad, back_idx, xyz, weight)
        wgrad = wgrad.sum(dim=0).reshape(*weight.shape)
        return None, None, wgrad, None

knn_spse = NSPSE.apply

class LASPSEA4(Function):
    @staticmethod
    # note  custom conversion saves memory i.e. only bf16 tensor for backward
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, feature: torch.Tensor, xyz: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor, training: bool=True) -> torch.Tensor:
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.int32)
        cutils.la_spse_a4_forward(output, back_idx, feature.float(), xyz.float(), knn, weight.float())
        if training or weight.requires_grad or feature.requires_grad:
            ctx.save_for_backward(back_idx, feature, xyz, weight)
        return output
    
    @staticmethod
    # @custom_bwd
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        back_idx, feature, xyz, weight = ctx.saved_tensors
        weight = weight.view(4, -1, 1).repeat(1, 1, 4).view(4, -1)
        f_grad = torch.zeros_like(grad)
        w_grad = torch.zeros(256, *weight.shape, device=xyz.device)
        cutils.la_spse_backward(f_grad, w_grad, grad, back_idx, feature.float(), xyz.float(), weight.float())
        w_grad = w_grad.sum(dim=0).view(4, -1, 4).sum(dim=2)
        return f_grad, None, None, w_grad, None

la_spse_a4 = LASPSEA4.apply