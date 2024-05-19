from utils import Hilbert
import os

hilbert_curve = Hilbert.hilbert_curve
hilbert_vis = Hilbert.hilbert_vis

max_level = 10
save_root = 'save/'
if not os.path.exists(save_root):
    os.mkdir(save_root)

for n in range(2,max_level+1):
    queue = [1] * (1<<(n*2))
    with open(save_root + 'hcurve2d_sort^'+str(n)+'_0.txt', 'w') as f:
        for i in range(1, (1<<n) + 1):
            for j in range(1, (1<<n) + 1):
                rank = hilbert_curve(n, i, j) - 1
                f.write(str(rank) + ' ')
                queue[rank] = j - 1  + (i - 1) * (1<<n)   

    queue_2d = [[z//(1<<n), z%(1<<n),] for z in queue]
    if n <= 4:
        hilbert_vis(queue_2d, save_root + 'hcurve2d^'+str(n)+'_0.png')

    with open(save_root + 'hcurve2d_queue^'+str(n)+'_0.txt', 'w') as f:
        for i in queue:
            f.write(str(i) + ' ') 
    f.close()


for n in range(2,max_level+1):
    queue = [1] * (1<<(n*2))
    with open(save_root +'hcurve2d_sort^'+str(n)+'_1.txt', 'w') as f:
        for i in range(1, (1<<n) + 1):
            for j in range(1, (1<<n) + 1):
                rank = (1 << (2*n)) - hilbert_curve(n, i, j) 
                f.write(str(rank) + ' ')
                queue[rank] = j - 1  + (i - 1) * (1<<n)

    queue_2d = [[z//(1<<n), z%(1<<n),] for z in queue]
    if n <= 4:
        hilbert_vis(queue_2d, save_root + 'hcurve2d^'+str(n)+'_1.png')

    with open(save_root +'hcurve2d_queue^'+str(n)+'_1.txt', 'w') as f:
        for i in queue:
            f.write(str(i) + ' ') 
    f.close()

for n in range(2,max_level+1):
    queue = [1] * (1<<(n*2))
    with open(save_root +'hcurve2d_sort^'+str(n)+'_2.txt', 'w') as f:
        for i in range(1, (1<<n) + 1):
            for j in range(1, (1<<n) + 1):
                rank =  hilbert_curve(n, j,i) - 1
                f.write(str(rank) + ' ')
                queue[rank] = j - 1  + (i - 1) * (1<<n)

    queue_2d = [[z//(1<<n), z%(1<<n),] for z in queue]
    if n <= 4:
        hilbert_vis(queue_2d, save_root + 'hcurve2d^'+str(n)+'_2.png')

    with open(save_root +'hcurve2d_queue^'+str(n)+'_2.txt', 'w') as f:
        for i in queue:
            f.write(str(i) + ' ') 
    f.close()

for n in range(2,max_level+1):
    queue = [1] * (1<<(n*2))
    with open(save_root +'hcurve2d_sort^'+str(n)+'_3.txt', 'w') as f:
        for i in range(1, (1<<n) + 1):
            for j in range(1, (1<<n) + 1):
                rank = (1 << (2*n)) - hilbert_curve(n, j, i) 
                f.write(str(rank) + ' ')
                queue[rank] = j - 1  + (i - 1) * (1<<n)

    queue_2d = [[z//(1<<n), z%(1<<n),] for z in queue]
    if n <= 4:
        hilbert_vis(queue_2d, save_root + 'hcurve2d^'+str(n)+'_3.png')

    with open(save_root +'hcurve2d_queue^'+str(n)+'_3.txt', 'w') as f:
        for i in queue:
            f.write(str(i) + ' ') 
    f.close()