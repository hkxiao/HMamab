iou1 = []
with open('48.3iou.txt', 'r') as f:
    for line in f.readlines():
        print(line,line.split(' | ')[-1])
        iou = float(line.split(' | ')[-1][:-2])
        iou1.append(iou)
        
iou2 = []
with open('47.9iou.txt', 'r') as f:
    for line in f.readlines():
        iou = float(line.split(' | ')[-1][:-2])
        iou2.append(iou)

print(len(iou1), len(iou2))
with open('weight.txt','w') as f:
    for i in range(150):
        if iou2[i] >= iou1[i]:
            f.write('1')
        elif iou1[i] - iou2[i] < 1.0:
            f.write('1.1')
        elif iou1[i] - iou2[i] < 2.0:
            f.write('1.2')
        elif iou1[i] - iou2[i] < 3.0:
            f.write('1.3')
        elif iou1[i] - iou2[i] < 4.0:
            f.write('1.4') 
        else:
            f.write('1.5') 
        f.write(',\n')
    