import os
import argparse
import torch
from torch.nn import functional as F
import numpy as np
from skimage.io import imread, imsave
import imageio
from tqdm import tqdm

from utils.pytorch_msssim import msssim, ssim
from utils.vis_tools import visualize_gray

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='output/walking_white_faced_egret')
args = parser.parse_args()

imgs_path = os.path.join(args.root_path, 'data')
os.makedirs(os.path.join(imgs_path, 'conf_map'), exist_ok=True)
img_files = os.listdir(imgs_path)
img_files.sort()

image_size = 256
view_num = 16

imgs = []
for img in img_files:
    if '.png' in img:
        imgs.append(imread(os.path.join(imgs_path, img)))

frame_num = len(imgs)

from model.RIFE_HDv3 import Model
model = Model()
model.load_model('model', -1)
print("Loaded v3.x HD model.")
model.eval()
model.device()


def warp_img(rgb1, rgb2):
    n, c, h, w = rgb1.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(rgb1, padding)
    img1 = F.pad(rgb2, padding)

    warped_img = model.inference(img0, img1)
    return warped_img


def get_conf_map(rgb_prev, rgb_next, rgb):
    rgb_prev = torch.tensor(rgb_prev, dtype=torch.float).cuda()
    rgb_next = torch.tensor(rgb_next, dtype=torch.float).cuda()
    rgb1 = rgb_prev[None, ...].permute(0, 3, 1, 2) / 255
    rgb2 = rgb_next[None, ...].permute(0, 3, 1, 2) / 255
    rgb = rgb[None, ...].permute(0, 3, 1, 2) / 255

    warped_rgb = warp_img(rgb1, rgb2)
    ssim_map = ssim(rgb, warped_rgb, full=True)
    ssim_err_map = (1 - ssim_map.mean(1))[0]

    rgb_err_map = (rgb - warped_rgb).abs()[0].sum(0)

    return ssim_err_map, rgb_err_map


for frame_i in tqdm(range(frame_num)):
    ssim_maps = []
    rgb_maps = []
    for view_i in range(view_num):
        rgb = np.copy(imgs[frame_i][:, view_i * image_size:(view_i + 1) * image_size, :])
        rgb = torch.tensor(rgb, dtype=torch.float).cuda()

        if frame_i == 0 or frame_i == frame_num-1:
            ssim_conf = torch.ones(256, 256)
            rgb_conf = torch.ones(256, 256)
        elif frame_i == 1 or frame_i == frame_num-2:
            rgb_prev = np.copy(imgs[frame_i - 1][:, view_i * image_size:(view_i + 1) * image_size, :])
            rgb_next = np.copy(imgs[frame_i + 1][:, view_i * image_size:(view_i + 1) * image_size, :])
            ssim_err_map, rgb_err_map = get_conf_map(rgb_prev, rgb_next, rgb)
            ssim_conf = ssim_err_map.max() - ssim_err_map
            rgb_conf = rgb_err_map.max() - rgb_err_map
        else:
            rgb_prev = np.copy(imgs[frame_i - 1][:, view_i * image_size:(view_i + 1) * image_size, :])
            rgb_next = np.copy(imgs[frame_i + 1][:, view_i * image_size:(view_i + 1) * image_size, :])
            ssim_err_map1, rgb_err_map1 = get_conf_map(rgb_prev, rgb_next, rgb)

            rgb_prev = np.copy(imgs[frame_i - 2][:, view_i * image_size:(view_i + 1) * image_size, :])
            rgb_next = np.copy(imgs[frame_i + 2][:, view_i * image_size:(view_i + 1) * image_size, :])
            ssim_err_map2, rgb_err_map2 = get_conf_map(rgb_prev, rgb_next, rgb)

            ssim_err_map = 0.5 * (ssim_err_map1 + ssim_err_map2)
            rgb_err_map = 0.5 * (rgb_err_map1 + rgb_err_map2)
            ssim_conf = ssim_err_map.max() - ssim_err_map
            rgb_conf = rgb_err_map.max() - rgb_err_map

        ssim_maps.append(torch.clamp(ssim_conf / ssim_conf.max(), 0, 1))
        rgb_maps.append(torch.clamp(rgb_conf / rgb_conf.max(), 0, 1))

        # conf_color = visualize_gray(ssim_err_map.detach().cpu().numpy())
        # imageio.imwrite(f'./conf_map.png', conf_color)

    ssim_maps = (torch.cat(ssim_maps, dim=1).detach().cpu().numpy() * 255).astype(np.uint8)
    imsave(os.path.join(imgs_path, 'conf_map', f'{frame_i:02d}_ssim.png'), ssim_maps)

    rgb_maps = (torch.cat(rgb_maps, dim=1).detach().cpu().numpy() * 255).astype(np.uint8)
    imsave(os.path.join(imgs_path, 'conf_map', f'{frame_i:02d}_rgb.png'), rgb_maps)
