import matplotlib.pyplot as plt
import random
import torch

def swap(a, b):
    return b, a

def partition(n, m, dir):
    n1, n2 = n // 2, n - n//2
    m1, m2 = m // 2, m - m//2

    if dir == 'a' or dir == 'b':
        if n1 % 2: n1, n2 = swap(n1,n2)
        if m1 % 2: m1, m2 = swap(m1,m2) 
    else:
        if n2 % 2: n1, n2 = swap(n1,n2)
        if m2 % 2: m1, m2 = swap(m1,m2) 
    return n1, n2, m1, m2

class Hilbert:
    def __init__():
        pass
    
    @staticmethod
    def hilbert_curve(n, m, dir, x, y):
        if max(n,m)<=2:
            if max(n,m) ==1: return 0
            if min(n,m) ==2:
                if dir== 'a': return [[0,0],[0,1],[1,1],[1,0]].index([x,y]) 
                if dir== 'b': return [[0,0],[1,0],[1,1],[0,1]].index([x,y]) 
                if dir== 'c': return [[1,1],[1,0],[0,0],[0,1]].index([x,y]) 
                if dir== 'd': return [[1,1],[0,1],[0,0],[1,0]].index([x,y]) 
            else:
                try:
                    if dir == 'a': return [[0,0],[1,0]].index([x,y]) 
                    if dir == 'b': return [[0,0],[0,1]].index([x,y]) 
                    if dir == 'c': return [[1,0],[0,0]].index([x,y])
                    if dir == 'd': return [[0,1],[0,0]].index([x,y]) 
                except:
                    print(dir,x,y,n,m)
                    print(0)  
                    return random.randint(0,1)
                # if n==2:
                #     if dir <= 'b':return [[0,0],[0,1]].index([x,y])  
                #     return [[0,1],[0,0]].index([x,y])
                # else:
                #     if dir <= 'b':return [[0,0],[1,0]].index([x,y])  
                #     return [[1,0],[0,0]].index([x,y])
        n1,n2,m1,m2 =  partition(n,m,dir)
        
        # print(n1,n2,m1,m2)
        # raise NameError
        if dir == 'a': dir1,dir2,dir3,dir4 = 'b','d', 'a', 'a'
        if dir == 'b': dir1,dir2,dir3,dir4 = 'a','b', 'c', 'b'
        if dir == 'c': dir1,dir2,dir3,dir4 = 'c','c', 'b', 'd'
        if dir == 'd': dir1,dir2,dir3,dir4 = 'd','a', 'd', 'c'
        
        if x < m1 and y < n1:
            if dir <= 'b': min_index = 0
            elif dir == 'c': min_index = n*m2
            else: min_index = n2*m
            
            return Hilbert.hilbert_curve(n1,m1,dir1,x,y)+min_index
        elif x >= m1 and y<n1:
            if dir == 'a': min_index = n*m1+n2*m2
            elif dir == 'b': min_index = n1*m1
            elif dir == 'c': min_index = n2*m2
            else: min_index = n2*m+n1*m1
            return Hilbert.hilbert_curve(n1,m2,dir2,x-m1,y)+min_index
        elif x < m1 and y>=n1:
            if dir == 'a': min_index = n1*m1
            elif dir == 'b': min_index = n1*m + n2*m2
            elif dir == 'c': min_index = n*m2 + n1*m1
            else: min_index = n2*m2
            return Hilbert.hilbert_curve(n2,m1,dir3,x,y-n1)+min_index
        else: 
            if dir == 'a': min_index = n*m1
            elif dir == 'b': min_index = n1*m
            else: min_index = 0
            return Hilbert.hilbert_curve(n2,m2,dir4,x-m1,y-n1)+min_index
        
    @staticmethod
    def hilbert_vis(curve, save_path='vis.png'):
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
        if clip==False: h_,w_=h,w
        
        print('start making hilbert curve ', h, w, 'flip' if flip==True else '', 'shift' if shift==True else '')
        queue = [1] * (h*w)
        for i in range(0, h):
            for j in range(0, w):
                rank = Hilbert.hilbert_curve(h, w, dir, j, h-i-1)
                queue[rank] = j+i*w   
 
        if clip:  queue = [x//w*w_ + x%w for x in queue if x//w<h_ and x%w<w_]       
        if shift: queue = [(x//w_+1)%w_*w_ + (x%w_+1)%w_  for x in queue]
        if flip: queue.reverse()
        
        queue_2d = [[z//w, z%w] for z in queue]    
        if max(h,w) <= 16:
            Hilbert.hilbert_vis(queue_2d, save_dir + '/hilbert_'+dir+'_'+ ('flip_'if flip==True else '')+''+('shift_'if shift==True else '')+str(h)+'_'+str(w)+'.jpg')
        
        queue_torch = torch.tensor(queue).cuda().int()
        sort_torch = torch.argsort(queue_torch)
        print('ending making hilbert curve ', h, w)
        return queue_torch, sort_torch 
