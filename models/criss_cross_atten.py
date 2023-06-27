import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
print("CC Moudle")

def INF(B,H,W,device):
     return -torch.diag(torch.tensor(float("inf"), device=device).repeat(H),0).unsqueeze(0).repeat(B*W,1,1) # float("inf")表示正无穷
    
class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: ([batch, c, h, w]) 例如([5, 64, 32, 16])
        device = x.device
        m_batchsize, _, height, width = x.size()

        proj_query = self.query_conv(x) # ([5, 8, 32, 16])
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1) # ([80, 32, 8])
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1) # ([160, 16, 8])

        proj_key = self.key_conv(x) # ([5, 8, 32, 16])
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) # ([80, 8, 32])
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) # ([160, 8, 16])

        proj_value = self.value_conv(x) # ([5, 64, 32, 16])
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) # ([80, 64, 32])
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) # ([160, 64, 16])

        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width, device=device)).view(m_batchsize,width,height,height).permute(0,2,1,3) # ([5, 32, 16, 32])
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width) # ([5, 32, 16, 16])

        concate = self.softmax(torch.cat([energy_H, energy_W], 3)) # ([5, 32, 16, 48])
        # print(concate.shape)

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height) # ([80, 32, 32])
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width) # ([160, 16, 16])
        # print(att_H.shape, att_W.shape) 

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1) # ([5, 64, 32, 16])
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3) # ([5, 64, 32, 16])
        # print(out_H.size(), out_W.size())
        return self.gamma*(out_H + out_W) + x


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.cca = CrissCrossAttention(inter_channels)
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, recurrence=2):
        residual = x
        output = self.conv1(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.conv2(output)
        output = self.conv3(torch.cat([x, output], 1))
        output = self.relu(residual + x)
        return output
    



if __name__ == '__main__':
    # model = CrissCrossAttention(64)
    model = RCCAModule(64, 64)
    x = torch.randn(5, 64, 32, 16)
    out = model(x) # ([5, 64, 32, 16])
    print(out.shape)
