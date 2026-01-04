
import torch.nn as nn
import torch
from tqdm import tqdm
class RQKmeans(nn.Module):
    def __init__(self,
                 cfg,
                ):
        super(RQKmeans, self).__init__()
        self.cfg = cfg
        self.num_book = cfg.model.num_book
        self.num_cluster = cfg.model.num_cluster
        self.dim = cfg.model.dim
        self.seed = cfg.seed
        self.device = cfg.device
        for i in range(self.num_book):
            self.register_buffer(f'codebook_{i}', torch.zeros(self.num_cluster[i], self.dim))
        


    def kmeans(self, input, K=1024, max_iter=50, atol=1e-5, rtol=0.0, generator=None):
        N, D = input.shape
        
        # Option A: Initialize from raw input to capture magnitude
        C = input[torch.randperm(N, generator=generator)[:K], :].clone()
        
        # Pre-compute normalized input for cosine distance calculation
        input_norm = torch.nn.functional.normalize(input, p=2, dim=-1)

        for _ in tqdm(range(max_iter), desc="KMeans Iterations"):
            # # Normalize C for distance calculation only
            # C_norm = torch.nn.functional.normalize(C, p=2, dim=-1)
            
            # dist = 1 - torch.mm(input_norm, C_norm.t())
            #p2 distance
            dist = torch.cdist(input, C, p=2)
        
            index = torch.argmin(dist, dim=-1)
            
            C_new = torch.zeros_like(C)
            counts = torch.bincount(index, minlength=K).clamp(min=1).unsqueeze(-1).to(C.dtype)
            
            # Update using raw input to preserve magnitude
            C_new.index_add_(0, index, input)
            C_new = C_new/counts
            
            # Do NOT normalize C_new

            if torch.allclose(C, C_new, atol=atol, rtol=rtol):
                C = C_new
                print("KMeans converged")
                break
            C = C_new

        # 统计码本利用率
        utilization = torch.unique(index).numel() / K
        # # 归一化码本
        # C = torch.nn.functional.normalize(C, p=2, dim=-1)

        return C, utilization
        
    @torch.no_grad()
    @staticmethod
    def nearest(input, C):
        # input_norm = torch.nn.functional.normalize(input, p=2, dim=-1)
        # C_norm = torch.nn.functional.normalize(C, p=2, dim=-1)
        # dist = 1 - torch.mm(input_norm, C_norm.t())
        dist = torch.cdist(input, C, p=2)
        index = torch.argmin(dist, dim=-1)
        return index
    
    @torch.no_grad()
    def fit(self, input):
        assert input.dim() == 2, "Input tensor must be 2-dimensional"

        residual = input.clone()
        utilizations = []
    
        for b in range(self.num_book):
            C, utilization = self.kmeans(residual, K=self.num_cluster[b], max_iter=50, atol=1e-5, rtol=0.0,
                            generator=torch.Generator().manual_seed(self.seed + b))
            utilizations.append(utilization)
            getattr(self, f'codebook_{b}').data.copy_(C.to(self.device))
            current_codebook = getattr(self, f'codebook_{b}')
            residual = residual - current_codebook[RQKmeans.nearest(residual, current_codebook)]
        
        print("Codebook utilizations:", utilizations)






    @torch.no_grad()
    def forward(self, input):
        assert input.dim() == 2, "Input tensor must be 2-dimensional"
        # assert self.codebooks, 
        output = []
        for i in range(self.num_book):
            current_codebook = getattr(self, f'codebook_{i}')
            idx = RQKmeans.nearest(input, current_codebook)
            input = input - current_codebook[idx]
            output.append(idx)

        output = torch.stack(output, dim=-1)
        res = input
        return output, res

        
        

    @torch.no_grad()
    def decode(self, sid):
        assert sid.dim() == 2, "Input tensor must be 2-dimensional"
        output = []
        for i in range(self.num_book):
            codebook = getattr(self, f'codebook_{i}')
            output.append(codebook[sid[:, i]])
        return torch.stack(output, dim=1)