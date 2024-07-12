import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import generate_spse_matrix, prepare_m, knn_spse, knn_spse_4, init_coor, la_spse_a4

all_dist = [[] for _ in range(10)]
spse_m = None

def checkpoint(function, *args, **kwargs):
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)

class LFP(nn.Module):
    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        out_dim //= 4
        self.coor = nn.Parameter(init_coor(out_dim).flatten())
        self.scale = nn.Parameter(torch.zeros(out_dim)+0.1)
        nn.init.constant_(self.bn.weight, init)

    def forward(self, x, knn):
        knn, xyz = knn
        B, N, C = x.shape
        x = self.proj(x)
        w = torch.cat([self.coor.view(-1, 3).transpose(0, 1), self.scale.square().view(1, -1)])
        x = la_spse_a4(x, xyz, knn, w, self.training)
        x = self.bn(x.view(B*N, -1)).view(B, N, -1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_dim, mlp_ratio, bn_momentum, act, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        nn.init.constant_(self.mlp[-1].weight, init)
    
    def forward(self, x):
        B, N, C = x.shape
        x = self.mlp(x.view(B*N, -1)).view(B, N, -1)
        return x

class Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act):
        super().__init__()

        self.depth = depth
        self.lfps = nn.ModuleList([
            LFP(dim, dim, bn_momentum) for _ in range(depth)
        ])
        self.mlp = Mlp(dim, mlp_ratio, bn_momentum, act, 0.2)
        self.mlps = nn.ModuleList([
            Mlp(dim, mlp_ratio, bn_momentum, act) for _ in range(depth // 2)
        ])
        if isinstance(drop_path, list):
            drop_rates = drop_path
            self.dp = [dp > 0. for dp in drop_path]
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
            self.dp = [drop_path > 0.] * depth
        #print(drop_rates)
        self.drop_paths = nn.ModuleList([
            DropPath(dpr) for dpr in drop_rates
        ])
    
    def drop_path(self, x, i, pts):
        if not self.dp[i] or not self.training:
            return x
        return torch.cat([self.drop_paths[i](xx) for xx in torch.split(x, pts, dim=1)], dim=1)

    def forward(self, x, knn, pts=None):
        x = x + self.drop_path(self.mlp(x), 0, pts)
        for i in range(self.depth):
            x = x + self.drop_path(self.lfps[i](x, knn), i, pts)
            if i % 2 == 1:
                x = x + self.drop_path(self.mlps[i // 2](x), i, pts)
        return x


class Stage(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()

        self.depth = depth
        self.up_depth = len(args.depths) - 1

        self.first = first = depth == 0
        self.last = last = depth == self.up_depth

        self.k = args.ks[depth]

        self.cp = cp = args.use_cp
        cp_bn_momentum = args.cp_bn_momentum if cp else args.bn_momentum

        dim = args.dims[depth]
        nbr_out_dim = args.dims[0]
        self.nbr_bn = nn.BatchNorm1d(dim, momentum=args.bn_momentum)
        nn.init.constant_(self.nbr_bn.weight, 0.8 if first else 0.2)
        self.nbr_proj = nn.Sequential(
            nn.BatchNorm1d(nbr_out_dim, momentum=args.bn_momentum),
            nn.Linear(nbr_out_dim, nbr_out_dim*2),
            args.act(),
            nn.Linear(nbr_out_dim*2, dim, bias=False)
        )
        
        self.sp_dim = nbr_out_dim
        if self.depth <= 1:
            self.spse_m = nn.Parameter(generate_spse_matrix(nbr_out_dim, args.all_dist if self.depth == 1 else args.all_dist0).flatten())
        if first:
            height = torch.linspace(0, 1, nbr_out_dim)
            height = height[torch.randperm(nbr_out_dim)]
            self.spse_h = nn.Parameter(height)
            col = torch.rand(3 * nbr_out_dim)
            self.spse_c = nn.Parameter(col)

        if not first:
            in_dim = args.dims[depth - 1]
            self.lfp = LFP(in_dim, dim, args.bn_momentum, 0.3)
            self.skip_proj = nn.Sequential(
                nn.Linear(in_dim, dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            )
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.blk = Block(dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, cp_bn_momentum, args.act)
        self.drop = DropPath(args.head_drops[depth])
        self.postproj = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            nn.Linear(dim, args.head_dim, bias=False),
        )
        nn.init.constant_(self.postproj[0].weight, (args.dims[0] / dim) ** 0.5)

        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )

        if not last:
            self.sub_stage = Stage(args, depth + 1)
    
    def local_aggregation(self, x, knn, pts):
        x = x.unsqueeze(0)
        x = self.blk(x, knn, pts)
        x = x.squeeze(0)
        return x

    def forward(self, x, xyz, prev_knn, indices, pts_list):
        """
        x: N x C
        """
        # downsampling
        if not self.first:
            ids = indices.pop()
            xyz = xyz[ids]
            x = self.skip_proj(x)[ids] + self.lfp(x.unsqueeze(0), prev_knn).squeeze(0)[ids]

        knn = indices.pop()

        N, k = knn.shape
        # rel_k = torch.randint(self.k - 1, (N, 1), device=x.device)
        # rel_k = torch.gather(knn[:, 1:], 1, rel_k).squeeze(1)
        # rel_cor = (xyz[rel_k] - xyz)
        # rel_cor = rel_cor.view(-1, 3)[::[523, 131, 31, 17, 7][self.depth]].mul_(self.cor_std)
        # r = rel_cor.square().sum(dim=1).sqrt().flatten()
        # all_dist[self.depth].append(r)
        # if len(all_dist[0]) == 600:
        #     dist = torch.cat(all_dist[self.depth][1:]).flatten().sort()[0]
        #     torch.save(dist.cpu(), f"dist{self.depth}.pt")
        #     if self.last:
        #         exit()


        
        # spatial encoding
        lxyz = F.pad(xyz, (0, 1)) * self.cor_std
        nbr_knn = knn.unsqueeze(0).to(torch.int32)
        if self.first:
            ipt = torch.cat([lxyz, x], dim=1).view(1, -1, 8)
            m = prepare_m(self.spse_m).view(self.sp_dim, 12).transpose(0, 1).contiguous()
            m = torch.cat([m, self.spse_c.view(3, self.sp_dim), self.spse_h.view(1, self.sp_dim)], dim=0)
            nbr = knn_spse_4(ipt, nbr_knn, m, self.training).view(N, -1).sqrt()
        else:
            if self.depth == 1:
                global spse_m
                spse_m = prepare_m(self.spse_m).view(self.sp_dim, 12).transpose(0, 1).contiguous()
            ipt = lxyz.view(1, -1, 4)
            nbr = knn_spse(ipt, nbr_knn, spse_m, self.training).view(N, -1).sqrt()
            
        nbr = self.nbr_proj(nbr)
        nbr = self.nbr_bn(nbr)
        x = nbr if self.first else nbr + x

        # main block
        knn = (nbr_knn, lxyz.view(1, -1, 4))
        pts = pts_list.pop() if pts_list is not None else None
        x = checkpoint(self.local_aggregation, x, knn, pts) if self.training and self.cp else self.local_aggregation(x, knn, pts)

        # get subsequent feature maps
        if not self.last:
            sub_x, sub_c = self.sub_stage(x, xyz, knn, indices, pts_list)
        else:
            sub_x = sub_c = None
        
        # regularization
        if self.training:
            rel_k = torch.randint(self.k, (N, 1), device=x.device)
            rel_k = torch.gather(knn[0].long().squeeze(0), 1, rel_k).squeeze(1)
            rel_cor = (xyz[rel_k] - xyz)
            rel_cor.mul_(self.cor_std)
            # print(rel_cor.std(dim=0))
            rel_p = x[rel_k] - x
            rel_p = self.cor_head(rel_p)
            closs = F.mse_loss(rel_p, rel_cor)
            sub_c = sub_c + closs if sub_c is not None else closs

        # upsampling
        x = self.postproj(x)
        if not self.first:
            back_nn = indices[self.depth-1]
            x = x[back_nn]
        x = self.drop(x)
        sub_x = sub_x + x if sub_x is not None else x

        return sub_x, sub_c

class DelaSemSeg(nn.Module):
    def __init__(self, args):
        super().__init__()

        # bn momentum for checkpointed layers
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum)**0.5

        self.stage = Stage(args)

        hid_dim = args.head_dim
        out_dim = args.num_classes

        self.head = nn.Sequential(
            nn.BatchNorm1d(hid_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(hid_dim, out_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, x, indices, pts_list=None):
        indices = indices[:]
        x, closs = self.stage(x, xyz, None, indices, pts_list)
        if self.training:
            return self.head(x), closs
        return self.head(x)

