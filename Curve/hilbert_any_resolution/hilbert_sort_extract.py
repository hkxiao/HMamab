from utils import Hilbert
import os

hilbert_curve = Hilbert.hilbert_curve
hilbert_vis = Hilbert.hilbert_vis

save_root = 'save/'
if not os.path.exists(save_root):
    os.mkdir(save_root)

n,m=16,9

queue = [1] * (n*m)
with open(save_root + 'hilbert2d_sort^'+str(n)+'_'+str(m)+'_1.txt', 'w') as f:
    for i in range(0, n):
        for j in range(0, m):
            rank = hilbert_curve(n, m,'a',j, i)
            f.write(str(rank) + ' ')
            print(rank)
            queue[rank] = j + i * m   

queue_2d = [[z//m, z%m] for z in queue]
    
if n <= 256:
    hilbert_vis(queue_2d, save_root + 'hilbert2d^'+str(n)+'_'+str(m)+'_1.png')

with open(save_root + 'hilbert2d_queue^'+str(n)+'_'+str(m)+'_1.txt', 'w') as f:
    for i in queue:
        f.write(str(i) + ' ') 

with open(save_root + 'hilbert2d_queue_2d^'+str(n)+'_'+str(m)+'_1.txt', 'w') as f:
    for point in queue_2d:
        f.write(str(point) + ' ') 
        
f.close()



# for n in range(0,max_level+1):
#     queue = [1] * (1<<(n*2))
#     with open(save_root +'hilbert2d_sort^'+str(n)+'_3.txt', 'w') as f:
#         for i in range(1, (1<<n) + 1):
#             for j in range(1, (1<<n) + 1):
#                 rank = (1 << (2*n)) - hilbert_curve(n, i, (1<<n)+1-j) 
#                 f.write(str(rank) + ' ')
#                 queue[rank] = j - 1  + (i - 1) * (1<<n)

#     queue_2d = [[z//(1<<n), z%(1<<n),] for z in queue]
#     if n <= 4:
#         hilbert_vis(queue_2d, save_root + 'hilbert2d^'+str(n)+'_3.png')

#     with open(save_root +'hilbert2d_queue^'+str(n)+'_3.txt', 'w') as f:
#         for i in queue:
#             f.write(str(i) + ' ') 
#     f.close()

# for n in range(0,max_level+1):
#     queue = [1] * (1<<(n*2))
#     with open(save_root +'hilbert2d_sort^'+str(n)+'_2.txt', 'w') as f:
#         for i in range(1, (1<<n) + 1):
#             for j in range(1, (1<<n) + 1):
#                 rank = (1 << (2*n)) - hilbert_curve(n, j, i) 
#                 f.write(str(rank) + ' ')
#                 queue[rank] = j - 1  + (i - 1) * (1<<n)

#     queue_2d = [[z//(1<<n), z%(1<<n),] for z in queue]
#     if n <= 4:
#         hilbert_vis(queue_2d, save_root + 'hilbert2d^'+str(n)+'_2.png')

#     with open(save_root +'hilbert2d_queue^'+str(n)+'_2.txt', 'w') as f:
#         for i in queue:
#             f.write(str(i) + ' ') 
#     f.close()

# for n in range(0,max_level+1):
#     queue = [1] * (1<<(n*2))
#     with open(save_root +'hilbert2d_sort^'+str(n)+'_0.txt', 'w') as f:
#         for i in range(1, (1<<n) + 1):
#             for j in range(1, (1<<n) + 1):
#                 rank = (1 << (2*n)) - hilbert_curve(n, (1<<n)-j+1, (1<<n)-i+1) 
#                 f.write(str(rank) + ' ')
#                 queue[rank] = j - 1  + (i - 1) * (1<<n)

#     queue_2d = [[z//(1<<n), z%(1<<n),] for z in queue]
#     if n <= 4:
#         hilbert_vis(queue_2d, save_root + 'hilbert2d^'+str(n)+'_0.png')

#     with open(save_root +'hilbert2d_queue^'+str(n)+'_0.txt', 'w') as f:
#         for i in queue:
#             f.write(str(i) + ' ') 
#     f.close()