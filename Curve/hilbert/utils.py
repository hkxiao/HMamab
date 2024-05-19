import matplotlib.pyplot as plt

class Hilbert:
    @staticmethod
    def hilbert_curve(n, x, y):
        if n == 0:
            return 1
        m = 1 << (n - 1)
        if x <= m and y <= m:
            return Hilbert.hilbert_curve(n - 1, y, x)
        if x > m and y <= m:
            return 3 * m * m + Hilbert.hilbert_curve(n - 1, m - y + 1, 2 * m - x + 1)
        if x <= m and y > m:
            return m * m + Hilbert.hilbert_curve(n - 1, x, y - m)
        if x > m and y > m:
            return 2 * m * m + Hilbert.hilbert_curve(n - 1, x - m, y - m)
        
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
    
    def make_queue_sort(h, w, dir, flip=False):
        
        pass
