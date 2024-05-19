import matplotlib.pyplot as plt
from math import log2
import torch

class Hcurve:
    @staticmethod
    def hcurve(n, y, x):
        if n == 2:
            len = 1<<n
            queue = [14,10,9,13,12,8,4,0,1,5,6,2,3,7,11,15]
            return queue.index(x+(y-1)*len-1) + 1
        m = 1 << (n - 1)
        bias = 1 << (2*n - 3)
        if x <= m and y <= m:
            return Hcurve.hcurve(n - 1, y, x) + m*m + bias
        if x > m and y <= m:
            return 3*m*m - Hcurve.hcurve(n-1, y, m*2-x+1)+1 + bias
        if x <= m and y > m:
            return m*m - Hcurve.hcurve(n-1, m*2-y+1, x) + 1 + bias 
        if x > m and y > m:
            
            if x < y:
                return Hcurve.hcurve(n - 1, y-m, x-m)
            elif x > y: 
                return Hcurve.hcurve(n - 1, y-m, x-m)  + 3*m*m
            else:
                if (x-m)%2:
                    return Hcurve.hcurve(n - 1, y-m, x-m)
                else:
                    return Hcurve.hcurve(n - 1, y-m, x-m) + 3*m*m
        
    @staticmethod
    def hcurve_vis(curve, save_path='vis.png'):
        # 设置图像的分辨率为100dpi
        plt.figure(dpi=500)

        # 提取横纵坐标
        x = [a[1] for a in curve]
        y = [a[0] for a in curve]

        # 提取矩阵
        H, W = max(y)+1, max(x)+1
        mat = range(1, H*W+1)
        mat = [mat[i*W:i*W+W] for i in range(H)]

        # 可视化
        plt.plot(x, y, marker='o', color='red')
        plt.plot(x[0], y[0], marker='o', color='black')  # 将第一个点表示为黑色
        plt.imshow(mat, cmap='viridis', interpolation='none')  # 显示矩阵

        # 显示坐标
        for i, j in curve:
            plt.text(j, i, f'({i},{j})', ha='center', va='center', color='white')
        
        plt.savefig(save_path, dpi=500)
        plt.clf() 
    
    def make_queue_sort(h, w, dir, shift=False, flip=False, save_dir=None, clip=False, h_=0, w_=0):
        print('start making hcurve', h, w, 'flip' if flip==True else '', 'shift' if shift==True else '')
        
        if clip==False: h_,w_=h,w
        queue = [1] * (h*w)
        for i in range(0, h):
            for j in range(0, w):
                if dir == 'a':
                    y, x = i, j
                elif dir == 'b':
                    x, y = i, j
                else:
                    raise RuntimeError(f'Hcurve {dir} not found.')
                
                rank = Hcurve.hcurve(int(log2(h)), y+1, x+1)-1
                queue[rank] = x+y*w   

        if clip:  queue = [x//w*w_ + x%w for x in queue if x//w<h_ and x%w<w_]       
        if shift: queue = [(x//w_+1)%w_*w_ + (x%w_+1)%w_  for x in queue]
        if flip: queue.reverse()
        
        queue_2d = [[z//w, z%w] for z in queue]    
        if max(h,w) <= 16:
            Hcurve.hcurve_vis(queue_2d, save_dir + '/hcurve_'+dir+'_'+ ('flip_'if flip==True else '')+''+('shift_'if shift==True else '')+str(h)+'_'+str(w)+'.jpg')
        
        queue_torch = torch.tensor(queue).cuda().int()
        sort_torch = torch.argsort(queue_torch)
        print('ending making hilbert curve ', h, w)
        return queue_torch, sort_torch
