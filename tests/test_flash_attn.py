import torch
import torch_spyre
import math

import torch
import torch_spyre
import math

def flash(Q,K,V, block_size):
    B, H, L, D = Q.shape

    output = torch.zeros_like(Q)
    M = torch.full((B,H,L), float('-inf'), device=Q.device)
    denominator = torch.zeros((B,H,L), device=Q.device)

    scale = 1.0 / math.sqrt(D)

    for start in range(0, L, block_size):
        end = start + block_size

        K_block = K[:, :, start:end, :] # B, H, Block, D
        V_block = V[:, :, start:end, :] # B, H, Block, D
        K_block_T = K_block.transpose(-1, -2).contiguous()  # B, H, D, Block
        print(K_block_T)
        
        scores = torch.matmul(Q, K_block_T) * scale
        # Use torch.amax directly instead of max(dim=-1).values to avoid argmax decomposition
        # Flash Attention only needs the maximum values, not the indices
        max_running = torch.maximum(M, torch.amax(scores, dim=-1))
        exp_scores = torch.exp(scores - max_running.unsqueeze(-1))

        denominator = denominator* torch.exp(M - max_running) + exp_scores.sum(dim=-1)
        output = output* torch.exp(M - max_running).unsqueeze(-1) + torch.bmm(exp_scores.flatten(0,1), V_block.flatten(0,1)).unflatten(0, (B,H))

        M = max_running
    output = output / denominator.unsqueeze(-1)
    return output

compiled_flash = torch.compile(flash, dynamic=False)

if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, L, D = 1, 8, 256, 64
    block_size = 128

    Q = torch.randn(B, H, L, D, dtype=torch.float16).to('spyre')
    K = torch.randn(B, H, L, D, dtype=torch.float16).to('spyre')
    V = torch.randn(B, H, L, D, dtype=torch.float16).to('spyre')

    #Q = torch.randn(B, H, L, D, dtype=torch.float32).to('cpu')
    #K = torch.randn(B, H, L, D, dtype=torch.float32).to('cpu')
    #V = torch.randn(B, H, L, D, dtype=torch.float32).to('cpu')

    #ref = torch.nn.functional.scaled_dot_product_attention(Q,K,V)
    #print(f'Ref: {ref}')

    #out = flash(Q,K,V,block_size) #Eager
    out = compiled_flash(Q,K,V,block_size) 
    print(f'Out: {out}')
