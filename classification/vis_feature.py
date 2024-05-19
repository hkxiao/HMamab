from models import build_model
from config import get_config
import argparse
import torch
from pathlib import Path
import cv2

parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', default='configs/hssm/hmambav2_tiny_224.yaml')
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

pth = 'pretrained/vssm_tiny_0230_ckpt_epoch_262.pth'
model.load_state_dict(torch.load(pth))

img_paths = ['/data/tanglv/data/imagenet1k/val/n01443537/ILSVRC2012_val_00000236.JPEG','/data/tanglv/data/imagenet1k/val/n01443537/ILSVRC2012_val_00000994.JPEG']
sizes = [224, 256, 512, 1024]

save_dir = 'feat_vis/'+ pth.split('/')[-1].split('.')[0]
Path(save_dir).mkdir(parents=True, exist_ok=True)

for img_path in img_paths:
    for size in sizes:
        img = torch.tensor(cv2.imread(img_path)).unqueeze(0).permute(0,3,1,2).flip(1).cuda()
        feat = model(img)
        
        print(feat.shape)
        save_name = img_path.split('/')[-1].split('.')[0] + '_' + str(size) + '_' + 'jpg'
        
        cv2.imwrite(save_dir + '/' + save_name)
        
        
        
    