import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2

eps = 1e-4#防止分母为零

# ==================================
# ======== Gaussian filter =========
# ==================================
#该函数用于生成高斯滤波器的基础滤波器。
def gaussian_basis_filters(scale, gpu, k=3):
    std = torch.pow(2,scale) #计算标准差：std = 2^scale（指数关系）

    # Define the basis vector for the current scale
    filtersize = torch.ceil(k*std+0.5) #计算滤波器尺寸：k*std向上取整，保证奇数尺寸

    if torch.isnan(filtersize).any():
        raise ValueError("filtersize为nan值")

    x = torch.arange(start=-filtersize.item(), end=filtersize.item()+1)
    if gpu is not None: x = x.to(gpu); std = std.to(gpu)
    x = torch.meshgrid([x,x])

    # Calculate Gaussian filter base
    # Only exponent part of Gaussian function since it is normalized anyway
    g = torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    g = g / torch.sum(g)  # Normalize

    # Gaussian derivative dg/dx filter base
    dgdx = -x[0]/(std**3*2*math.pi)*torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    dgdx = dgdx / torch.sum(torch.abs(dgdx))  # Normalize

    # Gaussian derivative dg/dy filter base
    dgdy = -x[1]/(std**3*2*math.pi)*torch.exp(-(x[1]/std)**2/2)*torch.exp(-(x[0]/std)**2/2)
    dgdy = dgdy / torch.sum(torch.abs(dgdy))  # Normalize

    # Stack and expand dim
    basis_filter = torch.stack([g,dgdx,dgdy], dim=0)[:,None,:,:]

    return basis_filter

#该函数用于对输入的batch进行高斯滤波卷积操作。
def convolve_gaussian_filters(batch, scale):
    if torch.isnan(scale).any():
        raise ValueError("scale为nan值")

    E, El, Ell = torch.split(batch, 1, dim=1)
    E_out, El_out, Ell_out = [], [], []

    for s in range(len(scale)):
        # Convolve with Gaussian filters
        w = gaussian_basis_filters(scale=scale[s:s+1], gpu=batch.device).to(dtype=batch.dtype)  # KCHW

        # the padding here works as "same" for odd kernel sizes
        E_out.append(F.conv2d(input=E[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))
        El_out.append(F.conv2d(input=El[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))
        Ell_out.append(F.conv2d(input=Ell[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))

    return torch.cat(E_out), torch.cat(El_out), torch.cat(Ell_out)



# == Color invariant definitions ==


def hat_H(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    H_single = torch.atan(El / (Ell + eps))  # 单通道 H
    return H_single


def hat_S(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    return (El ** 2 + Ell ** 2) / (E ** 2 + eps)


def hat_Ww(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Wx = Ex / (E + eps)
    Wy = Ey / (E + eps)
    return Wx ** 2 + Wy ** 2


def hat_Wlw2(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Wlx = Elx / (E + eps)
    Wly = Ely / (E + eps)
    return Wlx ** 2 + Wly ** 2


def hat_Wllw2(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Wllx = Ellx / (E + eps)
    Wlly = Elly / (E + eps)
    return Wllx ** 2 + Wlly ** 2


# == Color invariant convolution ==


class PriorConv2d(nn.Module):
    def __init__(self, n_fea_middle, k=3, scale=0.0, ablation='no_Ww'):

        super(PriorConv2d, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.n_fea_middle = n_fea_middle

        # Constants
        self.gcm = torch.nn.Parameter(torch.tensor([[0.06, 0.63, 0.27], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]]))
        self.k = k

        # 基础通道数
        base_channels = n_fea_middle // 4  # 7
        self.conv_H = nn.Conv2d(1, base_channels, kernel_size=1, bias=True)
        self.conv_S = nn.Conv2d(1, base_channels, kernel_size=1, bias=True)
        self.conv_RGB = nn.Conv2d(3, base_channels, kernel_size=1, bias=True)
        self.conv_Ww = nn.Conv2d(1, base_channels, kernel_size=1, bias=True)

        # 调整层
        self.conv_adjust = nn.Conv2d(base_channels * 4, n_fea_middle, kernel_size=1, bias=True)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            torch.nn.Conv2d(16, 1, 3, padding=1)
        )
        
        #权重预测分支.引入一个动态权重生成模块，根据输入图像的特征预测每个先验的权重，使权重随着图像内容变化。
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 提取局部特征
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 进一步提取特征
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1),  # 输出4个权重通道
            nn.Softmax(dim=1)  # 在通道维度上归一化
        )
        self.saved_features = {}
        self.ablation = ablation  # 记录消融模式

    def forward(self, x):
        # Make sure scale does not explode: clamp to max abs value of 2.5
        # self.scale.data = torch.clamp(self.scale.data, min=-2.5, max=2.5)

        with torch.no_grad():
            max_RGB = torch.argmax(x, dim=1)
            min_RGB = torch.argmin(x, dim=1)

            x_ = torch.flip(x, dims=(1,))

            max_RGB_ = 2 - torch.argmax(x_, dim=1)
            min_RGB_ = 2 - torch.argmin(x_, dim=1)

            RGB_order = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
            RGB_order = RGB_order.scatter_(1, max_RGB.unsqueeze(1), 0.5, reduce='add')
            RGB_order = RGB_order.scatter_(1, max_RGB_.unsqueeze(1), 0.5, reduce='add')
            RGB_order = RGB_order.scatter_(1, min_RGB.unsqueeze(1), -0.5, reduce='add')
            RGB_order = RGB_order.scatter_(1, min_RGB_.unsqueeze(1), -0.5, reduce='add')

        scale = torch.mean(self.conv(x), dim=(1, 2, 3))
        scale = torch.clamp(scale, min=-2.5, max=2.5)

        # Measure E, El, Ell by Gaussian color model
        in_shape = x.shape  # bchw
        x = x.view((in_shape[:2] + (-1,)))  # flatten image
        x = torch.matmul(self.gcm.to(x.device, dtype=x.dtype), x)  # estimate E,El,Ell
        x = x.view((in_shape[0],) + (3,) + in_shape[2:])  # reshape to original image size

        E_out, El_out, Ell_out = convolve_gaussian_filters(x.float(), scale.float())

        # print("W1")
        E, Ex, Ey = torch.split(E_out, 1, dim=1)
        El = torch.split(El_out, 1, dim=1)[0]
        Ell = torch.split(Ell_out, 1, dim=1)[0]

        H = hat_H(E, Ex, Ey, El, None, None, Ell, None, None)
        S = torch.log(hat_S(E, Ex, Ey, El, None, None, Ell, None, None) + eps)
        Ww = torch.atan(hat_Ww(E, Ex, Ey, El, None, None, Ell, None, None))

        # 计算动态权重
        weights = self.weight_predictor(x)  # [b, 4]，对空间维度取均值

        # 独立变换
        H_fea = self.conv_H(H) 
        S_fea = self.conv_S(S)  
        RGB_fea = self.conv_RGB(RGB_order)  
        Ww_fea = self.conv_Ww(Ww)  
        
        
        # 应用动态权重
        H_fea_w = self.conv_H(H) * weights[:, 0:1, :, :]  # [b, base_channels, h, w] * [b, 1, h, w]
        S_fea_w = self.conv_S(S) * weights[:, 1:2, :, :]
        RGB_fea_w = self.conv_RGB(RGB_order) * weights[:, 2:3, :, :]
        Ww_fea_w = self.conv_Ww(Ww) * weights[:, 3:4, :, :]
        # 消融实验：将指定先验特征置零
        if self.ablation == 'no_H':
            H_fea_w = torch.zeros_like(H_fea_w)
        elif self.ablation == 'no_S':
            S_fea_w = torch.zeros_like(S_fea_w)
        elif self.ablation == 'no_RGB':
            RGB_fea_w = torch.zeros_like(RGB_fea_w)
        elif self.ablation == 'no_Ww':
            Ww_fea_w = torch.zeros_like(Ww_fea_w)

         # 拼接
        features = torch.cat([H_fea_w, S_fea_w, RGB_fea_w, Ww_fea_w], dim=1)  # [b, 40, h, w]
        features = self.conv_adjust(features)  # [b, 40, h, w]

        # 保存原始特征和权重
        self.saved_features = {
            'H_fea': H_fea,
            'S_fea': S_fea,
            'RGB_fea': RGB_fea,
            'RGB_order':RGB_order,
            'Ww_fea': Ww_fea,
            'H_fea_w': H_fea * weights[:, 0:1],
            'S_fea_w': S_fea * weights[:, 1:2],
            'RGB_fea_w': RGB_fea * weights[:, 2:3],
            'Ww_fea_w': Ww_fea * weights[:, 3:4],
            'weights': weights,
            'features': features  # 来自最后的拼接
        }
        return features




#QuadPriorformer架构
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    '''该函数用于生成截断正态分布的张量。'''
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    '''该函数用于生成截断正态分布的张量。
      type:(Tensor, float, float, float, float) -> Tensor'''
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    '''该函数用于初始化张量，根据指定的模式和分布计算方差，并使用相应的分布初始化张量。'''
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    '''该函数 lecun_normal_ 使用 variance_scaling_ 函数对输入的张量进行初始化。
    具体来说，它使用 fan_in 模式和 truncated_normal 分布来计算方差，并根据计算结果初始化张量。'''
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    '''输入数据上应用层归一化（Layer Normalization），然后再调用传入的函数 fn 进行处理。'''
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    '''该函数 shift_back 用于对输入的四维张量进行列偏移操作。具体步骤如下：
        获取输入张量的形状，计算下采样率。
        根据步长和下采样率调整步长值。
        遍历每个通道，根据调整后的步长对每一行进行偏移。
        返回处理后的张量。'''
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


    
class QP_MSA(nn.Module):
    def __init__(self, dim, heads, dim_head=40):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        #self.rescale定义了一个可学习的参数 rescale，用于缩放多头注意力机制中的每个头。
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim
        self.saved_attn = {}  # 新增：用于保存中间特征

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        #这段代码的功能是将输入的查询、键、值和注意力矩阵进行重排。
        # 具体来说，使用 rearrange 函数将这些张量从形状 (b, n, h * d) 转换为 (b, h, n, d)，
        # 其中 b 是批次大小，n 是序列长度，h 是头数，d 是每个头的维度。
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v_illu = v * illu_attn  # 新增行
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1) #位置编码
        out = out_c + out_p

        # 保存中间特征到字典
        self.saved_attn = {
            'x_in': x_in,
            'q_inp': q_inp.detach(),   # [b, h*w, c]
            'k_inp': k_inp.detach(),
            'v_inp': v_inp.detach(),
            'QK_attn': attn.detach(),  # [b, heads, n, n]
            'v_illu': v_illu.detach(),# [b, heads, n, d]
            'attn_output': out.detach()# [b, h, w, c]
        }
        return out

class SEBlock(nn.Module):
    #将 Squeeze-and-Excitation (SE) 块添加到 FeedForward 模块的最后卷积层后，
    #以增强模型对通道特征的重新校准能力。这在低光图像增强任务中可能特别有效，
    #因为它能帮助模型更好地处理不同光照条件下的特征。
    #SE 块的 reduction 参数默认设置为 16，您可以根据任务需求调整此值以平衡性能和计算成本。
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.conv_gate = nn.Conv2d(dim, dim * mult, 1, 1, bias=False)
        self.conv_value = nn.Conv2d(dim, dim * mult, 1, 1, bias=False)
        self.conv_out = nn.Conv2d(dim * mult, dim, 1, 1, bias=False)
        self.swish = nn.SiLU()

    def forward(self, x):
        """
        x: [b, h, w, c]
        return out: [b, h, w, c]
        """
        x = x.permute(0, 3, 1, 2)  # 从 [b, h, w, c] 转为 [b, c, h, w]
        gate = self.swish(self.conv_gate(x))  # 门控分支，使用 SiLU 激活
        value = self.conv_value(x)  # 值分支
        out = gate * value  # 门控机制：逐元素相乘
        out = self.conv_out(out)  # 投影回原始维度
        return out.permute(0, 2, 3, 1)  # 转回 [b, h, w, c]




class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, reduction=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
        self.se = SEBlock(dim, reduction)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        #将输入张量 x 的维度顺序进行调整，并通过连续内存布局优化后传递给网络层 self.net 进行前向传播
        out = self.se(out)
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(self, dim, heads, dim_head=40, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                QP_MSA(dim=dim, heads=heads, dim_head=dim_head),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea.permute(0, 2, 3, 1)) + x  # 残差连接
            x = ff(x) + x  # 前馈网络 + 残差连接
            self.saved_attn_features = attn.saved_attn  # 保存当前块的中间特征
        out = x.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
        return out

class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=40, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # 输入投影
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # 编码器
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, 
                     heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # 瓶颈
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, 
                              num_blocks=num_blocks[-1])

        # 解码器
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, 
                                  padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], 
                     dim_head=dim, heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # 输出投影
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # 激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        '''该方法用于初始化模型中的权重。
        对于线性层（nn.Linear），使用截断正态分布初始化权重，并将偏置初始化为0；
        对于层归一化层（nn.LayerNorm），将偏置初始化为0，权重初始化为1。'''
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            # 检查并上采样 illu_fea，使其与 fea 的空间尺寸一致
            if illu_fea.shape[2:] != fea.shape[2:]:
                illu_fea = F.interpolate(
                    illu_fea, size=fea.shape[2:], mode='bilinear', align_corners=False
                )
            fea = LeWinBlock(fea, illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class QuadPriorFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, level=2, ablation='no_Ww',
                 num_blocks=[1, 1, 1]):
        super(QuadPriorFormer_Single_Stage, self).__init__()
        self.PriorConv2d = PriorConv2d(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, 
                                level=level, num_blocks=num_blocks)
        self.test_mode = False  # 添加模式标志，分开测试和训练

    def forward(self, img):
        # img:        b,c=3,h,w
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        illu_fea = self.PriorConv2d(img)
        input_img = img
        output_img = self.denoiser(input_img, illu_fea)
        if self.test_mode:  # 仅在测试模式返回特征
            return {
                'output': output_img,
                'weights': self.PriorConv2d.saved_features['weights'],
                'H_fea': self.PriorConv2d.saved_features['H_fea'],
                'S_fea': self.PriorConv2d.saved_features['S_fea'],
                'RGB_fea': self.PriorConv2d.saved_features['RGB_fea'],
                'RGB_order':self.PriorConv2d.saved_features['RGB_order'],
                'Ww_fea': self.PriorConv2d.saved_features['Ww_fea'],
                'H_fea_w': self.PriorConv2d.saved_features['H_fea_w'],
                'S_fea_w': self.PriorConv2d.saved_features['S_fea_w'],
                'RGB_fea_w': self.PriorConv2d.saved_features['RGB_fea_w'],
                'Ww_fea_w': self.PriorConv2d.saved_features['Ww_fea_w'],
                'features': illu_fea,
                'QP_MSA_features': self.denoiser.encoder_layers[0][0].saved_attn_features,  # 假设第一个encoder层
             }
       
        else:
            return output_img  # 训练模式返回张量
        
class QuadPriorFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=3, 
                 num_blocks=[1, 1, 1]):
        super(QuadPriorFormer, self).__init__()
        self.stage = stage
        modules_body = [QuadPriorFormer_Single_Stage(in_channels=in_channels, 
                                                  out_channels=out_channels, 
                                                  n_feat=n_feat, level=2, 
                                                  num_blocks=num_blocks, 
                                                  )
                        for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        
    def set_test_mode(self, mode=True):
        """设置测试模式"""
        for module in self.body:
            if isinstance(module, QuadPriorFormer_Single_Stage):
                module.test_mode = mode
                
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        if self.body[0].test_mode:  # 测试模式
            outputs = []
            for stage in self.body:
                stage_out = stage(x)
                outputs.append(stage_out)
                x = stage_out['output']  # 传递output到下一阶段
            return {
                'final_output': outputs[-1]['output'],
                'stage_outputs': outputs
            }
        else:  # 训练模式
            return self.body(x)  # 正常前向传播