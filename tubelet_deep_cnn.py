import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TubeletEmbeddingDNN(nn.Module):
  def __init__(self,c=3,d_model=96,tubelet_size=(2,4,4)):
    #tubelet_size=(t,h,w)
    super(TubeletEmbeddingDNN,self).__init__()
    self.tubelet_size=tubelet_size
    t,h,w=self.tubelet_size
    self.d_model=d_model
    self.c=c
    self.dnn=nn.ModuleList([
      nn.Sequential(
        nn.Conv3d(1,8,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv3d(8,16,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool3d((1,1,1)),
        nn.Flatten(),
        nn.Linear(16,self.d_model)
      ) for _ in range(self.c)
    ])
  def forward(self,x):
    #x=(B,C,T,H,W)
    B,C,T,H,W=x.shape
    t,h,w=self.tubelet_size
    assert T % t == 0 and H % h == 0 and W % w == 0, "input dims must be divisible by tubelet size"
    T_new=T//t
    H_new=H//h
    W_new=W//w
    N=T_new*H_new*W_new
    out=[]
    for ch in range(self.c):
       x_ch=x[:,ch:ch+1]
       x_tubelet=x_ch.view(B,1,T_new,t,H_new,h,W_new,w)
       x_tubelet=x_tubelet.permute(0,2,4,6,1,3,5,7)
       x_tubelet=x_tubelet.reshape(-1,1,t,h,w)
       z=self.dnn[ch](x_tubelet)#(B*N,d_model)
       z=z.view(B,N,self.d_model)
       out.append(z)
    out=torch.stack(out,dim=1) #(B,C,N,D)
    return out


video = torch.randn(2, 3, 8, 32, 32) 
print("Video dim (B,C,T,H,W):   ",video.shape)
tubelet_embed = TubeletEmbeddingDNN(c=3,d_model=128, tubelet_size=(2, 4, 4))
out = tubelet_embed(video)  
print("Tubelet Embedding(B,C,T',D):   ",out.shape)