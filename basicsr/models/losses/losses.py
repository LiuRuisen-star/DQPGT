import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss
from basicsr.models.losses.loss_util import reduce_loss
from pytorch_ssim import SSIM
from torchvision.models import vgg16


_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   #把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

# 新增：定义一个组合 L1 + SSIM 的损失类（可选）
class L1SSIMLoss(nn.Module):
    """组合 L1 和 SSIM 的损失函数"""
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, alpha=0.3, reduction='mean'):
        super(L1SSIMLoss, self).__init__()
        self.l1_loss = L1Loss(loss_weight=l1_weight, reduction=reduction)  # L1 损失
        self.ssim_loss = SSIM(window_size=11, size_average=True)  # SSIM 损失
        self.alpha = alpha  # L1 和 SSIM 的权重平衡参数
        self.ssim_weight = ssim_weight

    def forward(self, pred, target, weight=None, **kwargs):
        # 计算 L1 损失
        l1 = self.l1_loss(pred, target, weight)
        # 计算 SSIM 损失（SSIM 返回值在 [0, 1]，1 表示完全相似，需转换为损失形式）
        ssim = self.ssim_loss(pred, target)
        total_loss = self.alpha * l1 + (1 - self.alpha) * self.ssim_weight * (1 - ssim)
        return total_loss    


    
class PerceptualLoss(nn.Module):
    """感知损失，使用 VGG16 的中间层特征"""
    def __init__(self, loss_weight=1.0, reduction='mean', layer='relu3_3'):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features
        if layer == 'relu3_3':
            self.vgg = vgg[:16].eval()  # relu3_3 对应 VGG16 的第 16 层
        elif layer == 'relu4_3':
            self.vgg = vgg[:23].eval()  # relu4_3 对应 VGG16 的第 23 层
        else:
            raise ValueError(f'Unsupported layer: {layer}')
        
        for param in self.vgg.parameters():
            param.requires_grad = False  # 冻结 VGG 参数
        
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        pred_fea = self.vgg(pred)
        target_fea = self.vgg(target)
        loss = F.mse_loss(pred_fea, target_fea, reduction='none')
        return self.loss_weight * reduce_loss(loss, self.reduction)
    
class L1PerceptualLoss(nn.Module):
    """组合 L1 Loss 和 Perceptual Loss 的损失函数"""
    def __init__(self, l1_weight=1.0, perc_weight=0.1, reduction='mean'):
        super(L1PerceptualLoss, self).__init__()
        self.l1_loss = L1Loss(loss_weight=l1_weight, reduction=reduction)
        self.perceptual_loss = PerceptualLoss(loss_weight=perc_weight, reduction=reduction)

    def forward(self, pred, target, weight=None):
        l1 = self.l1_loss(pred, target, weight)
        perc = self.perceptual_loss(pred, target)
        total_loss = l1 + perc
        return total_loss
    
    
class L1PerceptualSSIMLoss(nn.Module):
    """组合 L1 Loss、Perceptual Loss 和 SSIM Loss 的损失函数"""
    def __init__(self, l1_weight=1.0, perc_weight=0.1, ssim_weight=0.3, reduction='mean'):
        super(L1PerceptualSSIMLoss, self).__init__()
        self.l1_loss = L1Loss(loss_weight=l1_weight, reduction=reduction)  # L1 损失
        self.perceptual_loss = PerceptualLoss(loss_weight=perc_weight, reduction=reduction)  # 感知损失
        self.ssim_module = SSIM(window_size=11, size_average=True)  # SSIM 模块
        self.ssim_weight = ssim_weight  # SSIM 损失权重

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred (Tensor): 预测图像，形状 (N, 3, H, W)
            target (Tensor): 目标图像，形状 (N, 3, H, W)
            weight (Tensor, optional): 元素级权重，形状 (N, C, H, W)
        Returns:
            Tensor: 总损失（L1 + Perceptual + SSIM）
        """
        l1 = self.l1_loss(pred, target, weight)  # 计算 L1 损失
        perc = self.perceptual_loss(pred, target)  # 计算 Perceptual 损失
        ssim = self.ssim_module(pred, target)  # 计算 SSIM 值
        ssim_loss = 1 - ssim  # 转换为损失形式（1 表示完全相似）
        total_loss = l1 + perc + self.ssim_weight * ssim_loss  # 线性组合
        return total_loss   
    
    
    
    
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction
    
#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction
    

#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss




