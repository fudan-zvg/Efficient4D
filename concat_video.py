import os
import numpy as np
from skimage.io import imread, imsave
from moviepy.editor import ImageSequenceClip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='output/walking_white_faced_egret')
parser.add_argument('--num_frames', type=int, default=16)
parser.add_argument('--interp', action='store_true')
args = parser.parse_args()

root_path = args.root_path
interp_path = os.path.join(root_path, 'interp')
if args.interp and not os.path.exists(interp_path):
    print('no interpolation frames')
    args.interp = False

N = args.num_frames
image_size = 256
image_files = []
for i in range(N):
    img = imread(os.path.join(root_path, f'{i}.png'))
    image_files.append(img)

    if args.interp:
        if i == N-1:
            break

        for j in range(1, 4):
            img = imread(os.path.join(root_path, 'interp', f'{i}_{i+1}', f'img{j}.png'))
            image_files.append(img)

data_path = os.path.join(root_path, 'data')
os.makedirs(data_path, exist_ok=True)
for i in range(len(image_files)):
    imsave(os.path.join(data_path, f'{i:02d}.png'), image_files[i])

video_path = os.path.join(root_path, 'preview')
os.makedirs(video_path, exist_ok=True)
imgs_360 = []
for index in range(16):
    imgs = []
    for long_img in image_files:
        rgb = np.copy(long_img[:, index * image_size:(index + 1) * image_size, :])
        imgs.append(rgb)

    output_video = video_path + f'/view_{index:02d}.mp4'
    clip = ImageSequenceClip(imgs, fps=30)
    clip.write_videofile(output_video, codec='libx264')

    # 360 deg
    f_index = index % N
    long_img = image_files[f_index]

    rgb = np.copy(long_img[:, index * image_size:(index + 1) * image_size, :])
    imgs_360.append(rgb)

output_video = video_path + f'/view360.mp4'
clip = ImageSequenceClip(imgs_360, fps=8)
clip.write_videofile(output_video, codec='libx264')
