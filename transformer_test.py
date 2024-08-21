import torch.nn as nn
import torch

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer_model=nn.Transformer(nhead=2,num_encoder_layers=1)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = transformer_model(src, tgt)

    print(out)