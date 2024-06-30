import os
import numpy as np
from skimage.io import imread, imsave
from ldm.base_utils import read_pickle
import json
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='output/walking_white_faced_egret')
parser.add_argument('--output', type=str, default='recon_data/walking_white_faced_egret')
parser.add_argument('--confidence', action='store_true')
args = parser.parse_args()

input_path = os.path.join(args.input, 'data')
num_frames = len(os.listdir(input_path)) - 1

for i in tqdm(range(num_frames)):
    image_path = os.path.join(input_path, f'{i:02d}.png')
    if args.confidence:
        ssim_path = os.path.join(input_path, 'conf_map', f'{i:02d}_ssim.png')
        rgb_path = os.path.join(input_path, 'conf_map', f'{i:02d}_rgb.png')
    else:
        ssim_path = image_path
        rgb_path = image_path

    num_views = 16
    image_size = 256
    log_dir = args.output + f'/{i:02d}'
    os.makedirs(log_dir, exist_ok=True)

    K, azs, els, dists, poses = read_pickle(f'meta_info/camera-16.pkl')

    R, t = poses[:, :, :3], poses[:, :, 3:]
    R = -R.transpose(0, 2, 1)
    t = R @ t  # imn,3,3 @ imn,3,1
    poses[:, :, :3] = R
    poses[:, :, 3:] = t

    row_3 = np.zeros_like(poses[0, 0:1, :])
    row_3[0, 3] = 1.0

    img = imread(image_path)
    conf_ssim = imread(ssim_path)
    conf_rgb = imread(rgb_path)
    frames = []

    for index in range(num_views):  # 16
        rgb = np.copy(img[:, index * image_size:(index + 1) * image_size, :])
        imsave(f'{log_dir}/rgb-{index:02d}.png', rgb)
        conf_ssim_map = np.copy(conf_ssim[:, index * image_size:(index + 1) * image_size])
        imsave(f'{log_dir}/conf_ssim-{index:02d}.png', conf_ssim_map)
        conf_rgb_map = np.copy(conf_rgb[:, index * image_size:(index + 1) * image_size])
        imsave(f'{log_dir}/conf_rgb-{index:02d}.png', conf_rgb_map)

        K, pose = np.copy(K), poses[index]

        frames.append({
            'file_path': f'{i:02d}/rgb-{index:02d}.png',
            'conf_ssim': f'{i:02d}/conf_ssim-{index:02d}.png',
            'conf_rgb': f'{i:02d}/conf_rgb-{index:02d}.png',
            'transform_matrix': np.concatenate([pose, row_3], axis=0).tolist(),
        })

    K = K.tolist()
    fl_x, fl_y, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    out_train = {
        'fl_x': fl_x,
        'fl_y': fl_y,
        'cx': cx,
        'cy': cy,
        'frames': frames,
    }
    out_test = {
        'fl_x': fl_x,
        'fl_y': fl_y,
        'cx': cx,
        'cy': cy,
        'frames': frames[::5],
    }
    output_train_path = f'{log_dir}/transforms_train.json'
    print(f'[INFO] write {len(frames)} images to {output_train_path}')
    with open(output_train_path, 'w') as f:
        json.dump(out_train, f, indent=2)
    output_test_path = f'{log_dir}/transforms_test.json'
    print(f'[INFO] write {len(frames[::5])} images to {output_test_path}')
    with open(output_test_path, 'w') as f:
        json.dump(out_test, f, indent=2)

time_list = sorted(
    [dirname for dirname in os.listdir(args.output) if '.json' not in dirname and '.ply' not in dirname and '.pickle' not in dirname])

train_frames_list = []
test_frames_list = []

fps = 30
for time_dir in time_list:
    with open(os.path.join(args.output, time_dir, 'transforms_train.json')) as fp:
        contents_train = fp.read()
    meta_train = json.loads(contents_train)  # load data as dict
    frames_train = meta_train['frames']
    for frame in frames_train:
        frame['file_path'] = frame['file_path'].replace('images', time_dir)
        frame['time'] = int(time_dir.split('-')[-1]) / fps
    train_frames_list.extend(frames_train)

    with open(os.path.join(args.output, time_dir, 'transforms_test.json')) as fp:
        contents_test = fp.read()
    meta_test = json.loads(contents_test)  # load data as dict
    frames_test = meta_test['frames']
    for frame in frames_test:
        frame['file_path'] = frame['file_path'].replace('images', time_dir)
        frame['time'] = int(time_dir.split('-')[-1]) / fps
    test_frames_list.extend(frames_test)

fl_x, fl_y, cx, cy = meta_train['fl_x'], meta_train['fl_y'], meta_train['cx'], meta_train['cy']
out_train = {
    'fl_x': fl_x,
    'fl_y': fl_y,
    'cx': cx,
    'cy': cy,
    'frames': train_frames_list,
}
out_test = {
    'fl_x': fl_x,
    'fl_y': fl_y,
    'cx': cx,
    'cy': cy,
    'frames': test_frames_list,
}

output_train_path = os.path.join(args.output, 'transforms_train.json')
print(f'[INFO] write {len(train_frames_list)} images to {output_train_path}')
with open(output_train_path, 'w') as f:
    json.dump(out_train, f, indent=2)
output_test_path = os.path.join(args.output, 'transforms_test.json')
print(f'[INFO] write {len(test_frames_list[::5])} images to {output_test_path}')
with open(output_test_path, 'w') as f:
    json.dump(out_test, f, indent=2)

