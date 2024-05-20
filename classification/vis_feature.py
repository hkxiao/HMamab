from models import build_model
from config import get_config
import argparse
import torch
from pathlib import Path
import cv2
from torch.nn import functional as F
import numpy as np
import shutil

def feature_vis(feats, save_pth): # feaats形状: [b,c,h,w]
    output_shape = (256,256) # 输出形状
    channel_mean = torch.mean(feats,dim=1,keepdim=True) # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
    channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
    channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().numpy() # 四维压缩为二维
    channel_mean = (((channel_mean - np.min(channel_mean))/(np.max(channel_mean)-np.min(channel_mean)))*255).astype(np.uint8)
    channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)

    cv2.imwrite(save_pth,channel_mean)


parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', default='configs/vssm/vmambav2_tiny_224.yaml')
#parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', default='configs/hssm/hmambav2_tiny_224_zig_zag_mlp.yaml')

parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)

args, unparsed = parser.parse_known_args()
config = get_config(args)

# print(config)
# raise NameError

model = build_model(config)
model.cuda()

pth = '../segmentation/official_pretrained/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_iter_160000.pth'
#pth = '../segmentation/output/hcurve-hilbert-zigzag-absposem/iter_160000.pth'

_ckpt = torch.load(pth)
print(_ckpt.keys())
#incompatibleKeys = model.load_state_dict(_ckpt['model'], strict=False)

print(_ckpt['state_dict'].keys())
new_statedict={}
for k,v in _ckpt['state_dict'].items():
    if 'backbone' in k:
        new_statedict[k.replace('backbone.','')] = v
        

# print(new_statedict.keys())
incompatibleKeys = model.load_state_dict(new_statedict, strict=False)

print(incompatibleKeys)

img_paths = ['/data/tanglv/data/imagenet1k/val/n01443537/ILSVRC2012_val_00000236.JPEG','/data/tanglv/data/imagenet1k/val/n01443537/ILSVRC2012_val_00000994.JPEG','/data/tanglv/data/ADEChallengeData2016/images/validation/ADE_val_00000001.jpg']
sizes = [224, 256, 512, 1024]

save_dir = 'feat_vis/'+ pth.split('/')[-1].split('.')[0]
Path(save_dir).mkdir(parents=True, exist_ok=True)

for img_path in img_paths:
    shutil.copyfile(img_path, save_dir + '/' + img_path.split('/')[-1])

    for size in sizes:        
        img = torch.tensor(cv2.imread(img_path).astype(np.float32)).unsqueeze(0).permute(0,3,1,2).flip(1).cuda()
        img = F.interpolate(img, size=(size,size),align_corners=False,mode='bilinear')

        print(size)
        with torch.no_grad():
            feats = model.forward_feature(img)
                
        for dep, feat in enumerate(feats):
            save_name = save_dir + '/' + img_path.split('/')[-1].split('.')[0] + '_size_' + str(size) + '_dep_' + str(dep)+ '.jpg' 
            feature_vis(feat, save_name)       

        
        
        
    