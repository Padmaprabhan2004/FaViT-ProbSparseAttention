import torch
import torch.nn as nn
from torchvision.models.video import s3d, S3D_Weights

class FineGrainedS3DTubeletEmbedding(nn.Module):

    def __init__(self, embedding_dim=128, freeze_s3d=True):
        super().__init__()

        weights = S3D_Weights.DEFAULT
        full_model = s3d(weights=weights)

        # Up to Mixed_4f (inclusive) → output dim is 528
        self.backbone = nn.Sequential(*list(full_model.features.children())[:12])
        self.backbone_output_dim = 528  # actual output of Mixed_4f

        if freeze_s3d:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.per_channel_proj = nn.ModuleList([
            nn.Linear(self.backbone_output_dim, embedding_dim) for _ in range(3)
        ])

    def forward(self, x):
    
        B, C, T, H, W = x.shape
        features_per_channel = []

        for ch in range(C):
            x_ch=x[:,ch:ch+1, :, :, :]           
            x_ch_rgb=x_ch.repeat(1, 3, 1, 1, 1)      

            with torch.no_grad():
                feat = self.backbone(x_ch_rgb)        

            B_f, D, T_f, H_f, W_f = feat.shape
            feat = feat.permute(0, 2, 3, 4, 1).reshape(B_f, -1, D) 

            projected = self.per_channel_proj[ch](feat)  
            features_per_channel.append(projected.unsqueeze(1)) 

        return torch.cat(features_per_channel, dim=1)  

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable



if __name__ == "__main__":
    model = FineGrainedS3DTubeletEmbedding(embedding_dim=128).eval()
    x = torch.randn(2, 3, 32, 224, 224)  # (B, C, T, H, W)
    out = model(x)
    print(out.shape)  # Expected: (2, 3, N, 128)
    total,trainable=count_parameters(model)
    print("Total Parameters :",total)
    print("Trainable Parameters :",trainable)
