"""
Local Mamba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from math import ceil, log2
from Curve.hilbert_any_resolution.utils import Hilbert
from Curve.h_curve.utils import Hcurve
curve_queue_map = {}
curve_sort_map = {}
flag = False

class MultiScan(nn.Module):

    ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip', 'w2', 'w2_flip', 'w7', 'w7_flip', 'hilbert', 'hcurve')

    def __init__(self, dim, choices=None, token_size=(14, 14)):

        super().__init__()
        self.token_size = token_size
        if choices is None:
            self.choices = MultiScan.ALL_CHOICES
            self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in self.choices])
            self.weights = nn.Parameter(1e-3 * torch.randn(len(self.choices), 1, 1, 1))
            self._iter = 0
            self.logger = logging.getLogger()
            self.search = True
        else:
            self.choices = choices
            self.search = False

    def forward(self, xs):
        """
        Input @xs: [[B, L, D], ...]
        """
        if self.search:
            weights = self.weights.softmax(0)
            xs = [norm(x) for norm, x in zip(self.norms, xs)]
            xs = torch.stack(xs) * weights
            x = xs.sum(0)
            if self._iter % 200 == 0:
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    self.logger.info(str(weights.detach().view(-1).tolist()))
            self._iter += 1
        else:
            x = torch.stack(xs).sum(0)
        return x

    def multi_scan(self, x):
        """
        Input @x: shape [B, L, D]
        """
        xs = []
        # print(self.choices)
        for direction in self.choices:
            xs.append(self.scan(x, direction))
        return xs

    def multi_reverse(self, xs):
        # [[B D L],...] -> [[B D L],..]
        new_xs = []
        for x, direction in zip(xs, self.choices):
            new_xs.append(self.reverse(x, direction)) 
        return new_xs

    def scan(self, x, direction='h'):
        """
        Input @x: shape [B, L, D] or [B, C, H, W]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if type(H) == torch.Tensor:
            H , W = H.item(), W.item()
            
        if len(x.shape) == 3:
            if direction == 'h':
                return x.transpose(-2, -1)
            elif direction == 'h_flip':
                return x.transpose(-2, -1).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan(x, K, H, W, flip=flip)
                # return LocalScanTriton.apply(x.transpose(-2, -1), K, flip, H, W)
            else:
                raise RuntimeError(f'Direction {direction} not found.')
        elif len(x.shape) == 4:
            if direction == 'h':
                return x.flatten(2)
            elif direction == 'h_flip':
                return x.flatten(2).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan_bchw(x, K, H, W, flip=flip)
                # return LocalScanTriton.apply(x, K, flip, H, W).flatten(2)
            elif 'hilbert' in direction:
                if x.requires_grad==True:
                    k = direction+'_'+str(H)+'_'+str(W)
                    dir = direction.split('_')[1]
                    flip = 'flip' in direction
                    shift = 'shift' in direction
            
                    if k not in curve_queue_map.keys():
                        curve_queue_map[k], curve_sort_map[k] = Hilbert.make_queue_sort(H, W, dir=dir, flip=flip, shift=shift, save_dir='curve_vis')
                        
                    return x.flatten(2)[:,:,curve_queue_map[k]]
                else:
                    log_level = max(ceil(log2(H)), ceil(log2(W)))
                    HH, WW = 1<<log_level, 1<<log_level
                                        
                    k = direction+'_'+str(HH)+'_'+str(WW)+'_clip_'+str(H)+'_'+str(W)
                    dir = direction.split('_')[1]
                    flip = 'flip' in direction
                    shift = 'shift' in direction

                    if k not in curve_queue_map.keys():
                        curve_queue_map[k], curve_sort_map[k] = Hilbert.make_queue_sort(HH, WW, dir=dir, flip=flip, shift=shift, save_dir='curve_vis',clip=True,h_=H,w_=W)
                        
                    return x.flatten(2)[:,:,curve_queue_map[k]]
                    
                    
            elif 'hcurve' in direction:    
                if x.requires_grad==True:
                    k = direction+'_'+str(H)+'_'+str(W)
                    dir = direction.split('_')[1]
                    flip = 'flip' in direction
                    shift = 'shift' in direction
            
                    if k not in curve_queue_map.keys():
                        curve_queue_map[k], curve_sort_map[k] = Hcurve.make_queue_sort(H, W, dir=dir, flip=flip, shift=shift, save_dir='curve_vis')
                        
                    return x.flatten(2)[:,:,curve_queue_map[k]]
                else:
                    log_level = max(ceil(log2(H)), ceil(log2(W)))
                    HH, WW = 1<<log_level, 1<<log_level
                                        
                    k = direction+'_'+str(HH)+'_'+str(WW)+'_clip_'+str(H)+'_'+str(W)
                    dir = direction.split('_')[1]
                    flip = 'flip' in direction
                    shift = 'shift' in direction

                    if k not in curve_queue_map.keys():
                        curve_queue_map[k], curve_sort_map[k] = Hcurve.make_queue_sort(HH, WW, dir=dir, flip=flip, shift=shift, save_dir='curve_vis',clip=True,h_=H,w_=W)
                        
                    return x.flatten(2)[:,:,curve_queue_map[k]]
                    
            
            else:
                raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h'):
        """
        Input @x: shape [B, D, L]
        Return torch.Tensor: shape [B, D, L]
        """
        
        #print(x.shape)
        H, W = self.token_size
        B, D, L = x.shape
        
        if type(H) == torch.Tensor:
            H , W = H.item(), W.item()
        
        # print(direction)
        
        if direction == 'h':
            # print(x[0,3,3])
            
            return x
        elif direction == 'h_flip':
            return x.flip([-1])
        elif direction == 'v':
            return x.view(B,D,W,H).permute(0,1,3,2).flatten(-2)
            return rearrange(x, 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction == 'v_flip':
            return x.view(B,D,W,H).permute(0,1,3,2).flatten(-2).flip(-1)
            return rearrange(x.flip([-1]), 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction.startswith('w'):
            K = int(direction[1:].split('_')[0])
            flip = direction.endswith('flip')
            return local_reverse(x, K, H, W, flip=flip)
            # return LocalReverseTriton.apply(x, K, flip, H, W)
        elif 'hilbert' in direction or 'hcurve' in direction:
            if x.requires_grad==True:
                k = direction+'_'+str(H)+'_'+str(W)
            else:
                log_level = max(ceil(log2(H)), ceil(log2(W)))
                HH, WW = 1<<log_level, 1<<log_level
                k = direction+'_'+str(HH)+'_'+str(WW)+'_clip_'+str(H)+'_'+str(W)
            return x[:,:,curve_sort_map[k]]
        else:
            raise RuntimeError(f'Direction {direction} not found.')    
        
    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace(self.__class__.__name__, f'{self.__class__.__name__}[{scans}]')

class MultiScanVSSM(MultiScan):

    ALL_CHOICES = MultiScan.ALL_CHOICES

    def __init__(self, dim, choices=None, sc_attn=True):
        super().__init__(dim, choices=choices, token_size=None)
        self.sc_attn = sc_attn
        if self.sc_attn:
            self.attn = BiAttn(dim)

    def merge(self, xs):
        # xs: [B, K, D, L]
        # return: [B, D, L]

        # remove the padded tokens
        xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
        # [B, K, D, L] -> [[B D L],...]
        
        xs = super().multi_reverse(xs) #[[B D L],...]
        
        if self.sc_attn:
            global flag
            if flag == False and xs[0].requires_grad == True:
                flag = True
                self.attn._init_weights()
            
            xs = [x.transpose(-2, -1) + self.attn(x.transpose(-2, -1)) for x in xs]
            
        x = super().forward(xs)
        return x

    
    def multi_scan(self, x):
        # x: [B, C, H, W]
        # return: [B, K, C, H * W]
        B, C, H, W = x.shape
        self.token_size = (H, W)

        xs = super().multi_scan(x)  # [[B, C, H, W], ...]

        self.scan_lengths = [x.shape[2] for x in xs]
        max_length = max(self.scan_lengths)

        # pad the tokens into the same length as VMamba compute all directions together
        new_xs = []
        for x in xs:
            if x.shape[2] < max_length:
                x = F.pad(x, (0, max_length - x.shape[2]))
            new_xs.append(x)
        return torch.stack(new_xs, 1)

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace('MultiScanVSSM', f'MultiScanVSSM[{scans}]')


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()
        
        # fusion
        self.fusion =  nn.Linear(in_channels, in_channels)

    def _init_weights(self):
        # 对所有子模块的参数进行初始化
        for module in self.modules():
            if hasattr(module, 'weight'):
                nn.init.zeros_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
                    
    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        # s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        # s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn #* s_attn  # [B, N, C]
        out = ori_x * attn
        
        return self.fusion(out)


