import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TubeletEmbedding(nn.Module):
  def __init__(self,c=3,d_model=96,tubelet_size=(2,4,4)):
    #tubelet_size=(t,h,w)
    super(TubeletEmbedding,self).__init__()
    self.tubelet_size=tubelet_size
    t,h,w=self.tubelet_size
    self.d_model=d_model
    self.lin_proj=nn.Linear(t*h*w,d_model)
  def forward(self,x):
    B,C,T,H,W=x.shape
    t,h,w=self.tubelet_size
    assert T % t == 0 and H % h == 0 and W % w == 0, "input dims must be divisible by tubelet size"
    T_new=T//t
    H_new=H//h
    W_new=W//w
    x=x.view(B,C,T_new,t,H_new,h,W_new,w)
    x=x.permute(0,1,2,4,6,3,5,7)
    x=x.contiguous().view(B,C,T_new*H_new*W_new,t*h*w)#(B,C,T,flattened_tubelet_dim)
    x=self.lin_proj(x)#(B,C,T,d_model)
    return x
  


class TubeletEmbeddingOverlapping(nn.Module):
    def __init__(self,c=3,d_model=96,tubelet_size=(2,4,4),stride=(1,1,1)):
        super(TubeletEmbeddingOverlapping,self).__init__()
        self.tubelet_size=tubelet_size
        self.stride=stride
        self.c=c
        self.d_model=d_model
        self.proj=nn.Conv3d(in_channels=self.c,out_channels=self.c*d_model,
                            kernel_size=self.tubelet_size,stride=self.stride,
                            padding=0,groups=c)
    def forward(self,x):
        B,C,T,H,W=x.shape
        x=self.proj(x)
        D=self.d_model
        T_new,H_new,W_new=x.shape[2:]
        x=x.view(B,C,D,T_new,H_new,W_new)
        x=x.permute(0,1,3,4,5,2)
        x=x.reshape(B,C,T_new*H_new*W_new,D)
        return x


video = torch.randn(2, 3, 8, 32, 32) 
print("Video dim (B,C,T,H,W):   ",video.shape)
tubelet_embed = TubeletEmbedding(c=3,d_model=128, tubelet_size=(2, 4, 4))
tubelet_embed_overlap=TubeletEmbeddingOverlapping(c=3,d_model=128,tubelet_size=(2,4,4),stride=(2,2,2))
out_overlap=tubelet_embed_overlap(video)
out = tubelet_embed(video)  
print("Tubelet Embedding(B,C,T',D):   ",out.shape)
print("Overlapping Tubelets(B,C,T'',D):   ",out_overlap.shape)