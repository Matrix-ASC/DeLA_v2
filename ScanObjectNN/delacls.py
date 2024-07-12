import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from pathlib import Path
import sys, math
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import init_coor, generate_spse_matrix, prepare_m, knn_spse, la_spse_a4
from pointnet2_ops import pointnet2_utils
from torch.cuda.amp import autocast

all_dist = [[] for _ in range(10)]
spse_m = None

@autocast(False)
def calc_pwd(x):
    x2 = x.square().sum(dim=2, keepdim=True)
    return x2 + x2.transpose(1, 2) + torch.bmm(x, x.transpose(1,2).mul(-2))

def get_graph_feature(x, idx):
    B, N, C = x.shape
    k = idx.shape[-1]
    nbr = torch.gather(x, 1, idx.view(B, N*k, 1).expand(-1, -1, C)).view(B*N, k, C)
    x = x.view(B*N, 1, C).expand(-1, k, -1)
    return nbr-x

def get_nbr_feature(x, idx):
    B, N, k = idx.shape
    C = x.shape[-1]
    nbr = torch.gather(x, 1, idx.view(B, N*k, 1).expand(-1, -1, C)).view(B*N*k, C)
    return nbr

class LFP(nn.Module):
    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.flag = in_dim == out_dim
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
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
        self.drop_paths = nn.ModuleList([
            DropPath(dpr) for dpr in drop_rates
        ])
        print(drop_path)

    def forward(self, x, knn):
        x = x + self.drop_paths[0](self.mlp(x))
        for i in range(self.depth):
            x = x + self.drop_paths[i](self.lfps[i](x, knn))
            if i % 2 == 1:
                x = x + self.drop_paths[i](self.mlps[i // 2](x))
        return x


class Stage(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()

        self.depth = depth

        self.first = first = depth == 0
        self.last = last = depth == len(args.depths) - 1

        self.n = args.ns[depth]
        self.k = args.ks[depth]

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

        if first:
            self.spse_m = nn.Parameter(generate_spse_matrix(nbr_out_dim, args.all_dist).flatten())
        self.sp_dim = nbr_out_dim

        if not first:
            in_dim = args.dims[depth - 1]
            self.lfp = LFP(in_dim, dim, args.bn_momentum, 0.3)
            self.skip_proj = nn.Sequential(
                nn.Linear(in_dim, dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            )
            nn.init.constant_(self.skip_proj[1].weight, 0.3)
        
        self.blk = Block(dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, args.bn_momentum, args.act)

        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )

        if not last:
            self.sub_stage = Stage(args, depth + 1)
    
    def forward(self, x, xyz, prev_knn, pwd):
        """
        x: B x N x C
        """
        # downsampling
        if not self.first:
            xyz = xyz[:, :self.n].contiguous()
            B, N, C = x.shape
            x = self.skip_proj(x.view(B*N, C)).view(B, N, -1)[:, :self.n] + self.lfp(x, prev_knn)[:, :self.n]

        _, knn = pwd[:, :self.n, :self.n].topk(k=self.k, dim=-1, largest=False, sorted=False)

        # B, N, _ = xyz.shape
        # rel_k = torch.randint(1, self.k, (B, N, 1), device=xyz.device)
        # rel_k = torch.gather(knn, 2, rel_k)
        # rel_cor = get_graph_feature(xyz, rel_k).flatten(1).mul_(self.cor_std)[::[43, 11, 3, 2][self.depth]]
        # r = rel_cor.square().sum(dim=1).sqrt().flatten()
        # all_dist[self.depth].append(r)
        # if len(all_dist[0]) == 300:
        #     dist = torch.cat(all_dist[self.depth]).flatten().sort()[0]
        #     torch.save(dist.cpu(), f"dist{self.depth}.pt")
        #     if self.last:
        #         exit()

        # spatial encoding
        B, N, k = knn.shape
        if self.first:
            global spse_m
            spse_m = prepare_m(self.spse_m).view(self.sp_dim, 12).transpose(0, 1).contiguous()
        
        ipt = F.pad(xyz, (0, 1)) * self.cor_std
        nbr_knn = knn.to(torch.int32)
        nbr = knn_spse(ipt, nbr_knn, spse_m, self.training).sqrt()

        nbr = self.nbr_proj(nbr.view(B*N, -1)).view(B, N, -1)
        nbr = self.nbr_bn(nbr.view(B*N, -1)).view(B, N, -1)
        x = nbr  if self.first else nbr + x

        # main block
        knn = (nbr_knn, ipt)
        x = self.blk(x, knn)

        # next stage
        if not self.last:
            sub_x, sub_c = self.sub_stage(x, xyz, knn, pwd)
        else:
            sub_x = x
            sub_c = None
        
        # regularization
        if self.training:
            rel_k = torch.randint(self.k, (B, N, 1), device=x.device)
            rel_k = torch.gather(knn[0].long(), 2, rel_k)
            rel_cor = get_graph_feature(xyz, rel_k).flatten(1).mul_(self.cor_std)
            # print(rel_cor.std(dim=0))
            rel_p = get_graph_feature(x, rel_k).flatten(1)
            rel_p = self.cor_head(rel_p)
            closs = F.mse_loss(rel_p, rel_cor)
            sub_c = sub_c + closs if sub_c is not None else closs

        return sub_x, sub_c


class DelaCls(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.stage = Stage(args)

        in_dim = args.dims[-1]
        out_dim = args.num_classes

        self.head = nn.Sequential(
            nn.BatchNorm1d(in_dim * 2, momentum=args.bn_momentum),
            nn.Linear(in_dim*2, out_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, xyz):
        if not self.training:
            idx = pointnet2_utils.furthest_point_sample(xyz, 1024).long()
        else:
            # resample
            idx = pointnet2_utils.furthest_point_sample(xyz, 1200).long()[:, torch.randperm(1200, device=xyz.device)[:1024]]
        xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        pwd = calc_pwd(xyz)
        x, closs = self.stage(None, xyz, None, pwd)
        x = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)
        if self.training:
            return self.head(x), closs
        else:
            return self.head(x)

