import torch
from torch import nn
from torch import Tensor
import kornia as K

class Masked_MSE(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        self.mse_loss = torch.nn.MSELoss(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, loss_mask: Tensor) -> Tensor:
        input = input[:, 0, :, :, :]
        target = target[:, 0, :, :, :]
        b, h, w, d = loss_mask.size()
        ib, ih, iw, id = input.size()
        p_size = ih // h  # assuming a unifrom patch size
        input = input.view(ib, h, p_size, w, p_size, d, p_size)
        target = target.view(ib,h, p_size, w, p_size, d, p_size)
        input = input.permute(0, 1, 3, 5, 2, 4, 6)
        target = target.permute(0, 1, 3, 5, 2, 4, 6)
        input = torch.masked_select(input, (loss_mask >= 1)[:, :, :, :, None, None, None])
        target = torch.masked_select(target, (loss_mask >= 1)[:, :, :, :, None, None, None])

        return self.mse_loss(input, target)

class Zero_Balance_MSE(nn.Module):
    def __init__(self,zero_weight, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        self.zero_weight = zero_weight
        self.mse_loss1 = torch.nn.MSELoss(size_average, reduce, reduction)
        self.mse_loss2 = torch.nn.MSELoss(size_average, reduce, reduction)
        self.mse_loss_detach = torch.nn.MSELoss(size_average, reduce, reduction)
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        zero_mask = (target==0.0)
        input_z = torch.masked_select(input, zero_mask)
        target_z = torch.masked_select(target, zero_mask)
        input_uz = torch.masked_select(input, torch.logical_not(zero_mask))
        target_uz = torch.masked_select(target,torch.logical_not(zero_mask))
        z_ratio = float(input_z.numel())/float(input.numel())
        loss_compensation = self.mse_loss_detach(input.detach(), target.detach())
        loss = self.mse_loss1(input_z,target_z)*z_ratio*self.zero_weight+self.mse_loss2(input_uz,target_uz)*(1-z_ratio)
        loss = loss * (loss_compensation.detach()/loss.detach())
        return loss

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]],dtype=torch.float32)
        sobel_y = torch.tensor([[[1,2,1],[0,0,0,],[-1,-2,-1]]],dtype=torch.float32)
        conv2d_weight = torch.stack([sobel_x,sobel_y])
        self.sobel_ops = nn.Conv2d(1,2,3,bias=False)
        self.sobel_ops.weight.data = conv2d_weight
        self.sobel_ops.weight.requires_grad = False
    def forward(self,img):
        return self.sobel_ops(img)
    def forward_enhance(self,img):
        edge = self.sobel_ops(img)
        edge = torch.abs(edge)
        edge = edge/4.0
        edge = torch.clip(edge,1e-5,1.0) #clip to 1e-5, or the grad will be error
        edge = edge**0.5
        return edge
class Edge_MSE(nn.Module):
    def __init__(self, hw_size=512, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        self.sobel_ops = Sobel()
        self.hw_size = hw_size
        self.mse = torch.nn.MSELoss(size_average, reduce, reduction)
    def forward_vanilla(self, input: Tensor, target: Tensor) -> Tensor:
        img = target.view(-1, 1, self.hw_size, self.hw_size)
        edge_gt = self.sobel_ops(img)
        output_edge = self.sobel_ops(input.view(-1, 1, self.hw_size, self.hw_size))
        return self.mse(output_edge,edge_gt)
    def forward_vanilla_norm(self, input: Tensor, target: Tensor) -> Tensor:
        img = target.view(-1, 1, self.hw_size, self.hw_size)
        edge_gt = self.sobel_ops(img)/4.0
        output_edge = self.sobel_ops(input.view(-1, 1, self.hw_size, self.hw_size))/4.0
        return self.mse(output_edge,edge_gt)
    def forward_enhance_edge(self, input: Tensor, target: Tensor) -> Tensor:
        img = target.view(-1, 1, self.hw_size, self.hw_size)
        img = K.filters.median_blur(img, (3, 3))
        edge_gt = self.sobel_ops.forward_enhance(img)
        output_edge = self.sobel_ops.forward_enhance(input.view(-1, 1, self.hw_size, self.hw_size))
        print(torch.min(output_edge),torch.max(output_edge))
        return self.mse(output_edge, edge_gt)
    def forward_clip(self, input: Tensor, target: Tensor) -> Tensor:
        img = target.view(-1, 1, self.hw_size, self.hw_size)
        edge_gt = torch.clip(self.sobel_ops(img),0.0,1.0)

        output_edge = torch.clip(self.sobel_ops(input.view(-1, 1, self.hw_size, self.hw_size)),0.0,1.0)
        return self.mse(output_edge,edge_gt)
if __name__ == "__main__":
    a = Sobel()
