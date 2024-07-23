# Efficient4D: Fast Dynamic 3D Object Generation from a Single-view Video
### [[Paper]](http://arxiv.org/abs/2401.08742) | [[Project]](https://fudan-zvg.github.io/Efficient4D/)

> [**Efficient4D: Fast Dynamic 3D Object Generation from a Single-view Video**](),            
> Zijie Pan, Zeyu Yang, [Xiatian Zhu](https://surrey-uplab.github.io/), [Li Zhang](https://lzrobots.github.io)  
> **Arxiv preprint**

**Official implementation of "Efficient4D: Fast Dynamic 3D Object Generation from a Single-view Video".** 

## Pipeline
<img width="768" alt="photo" src="assets/pipeline.png">

## Image generation results


https://github.com/fudan-zvg/Efficient4D/assets/84657631/26440964-5bff-4b1d-b240-4ba6b9e62e69


## Install
```bash
conda create -n efficient4d python=3.9
conda activate efficient4d
pip install -r requirements.txt
```

### Pretrained checkpoints
Following [Syncdreamer](https://github.com/liuyuan-pal/SyncDreamer) to download [checkpoints](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/yuanly_connect_hku_hk/EjYHbCBnV-VPjBqNHdNulIABq9sYAEpSz4NPLDI72a85vw) under `./ckpt`.
### Data
We provide one example case under `./data`, please refer to [Consistent4D](https://github.com/yanqinJiang/Consistent4D) for more data.
### Tested environments
* Ubuntu 20 with torch 1.13 & CUDA 11.7 on eight A6000.

## Usage
```bash
# set path
DEVICE="0,1,2,3"
N_FRAME=16  # 4 frames per GPU, 40G memory
INPUT="./data/walking_white_faced_egret"
OUTPUT="./output/walking_white_faced_egret"
OUTPUT_RECON="./recon_data/walking_white_faced_egret"

# generate pseudo muti-view and muti-frame consistent images
CUDA_VISIBLE_DEVICES=$DEVICE python generate.py \
  --input $INPUT \
  --output $OUTPUT \
  --crop_size 200 \
  --elevation 0 \
  --frame_num $N_FRAME \
  --smooth_filter \
  --decomposed_sampling \
  --is_cyc 0 \
  --seed 3407

# frame interpolation (optional)
for((i=0; i<$[N_FRAME-1]; i++))
do
  IMG1="$OUTPUT/$i.png"
  IMG2="$OUTPUT/$[i+1].png"
  SAVE_PATH="$OUTPUT/interp/${i}_$[i+1]"
  CUDA_VISIBLE_DEVICES=$DEVICE python frame_interpolation.py --img $IMG1 $IMG2 --cycle2 --save_path $SAVE_PATH
done

# preview
CUDA_VISIBLE_DEVICES=$DEVICE python concat_video.py --root_path $OUTPUT --num_frame $N_FRAME --interp

# confidence map
CUDA_VISIBLE_DEVICES=$DEVICE python video_confidence.py --root_path $OUTPUT

# pose
CUDA_VISIBLE_DEVICES=$DEVICE python data_convertor.py --input $OUTPUT --output $OUTPUT_RECON --confidence
```

We also provide a script `run.sh` to finish the above process in one command:
```bash
# bash run.sh DATA_NAME OUTPUT_NAME DEVICES N_FRAMES CYCLE
# We can set CYCLE=1 to enhance consistency if the input video is periodic 
# or the first frame and the last frame of the input video are similar.
bash run.sh walking_white_faced_egret walking_white_faced_egret 4,5,6,7 16 0
```

### 4D reconstruction
Now the preview videos are under `output/DATA_NAME/preview` and the data for reconstruction are under `./recon_data/DATA_NAME`, please following [4DGS](https://github.com/fudan-zvg/4d-gaussian-splatting) for reconstruction. 
Currently, the confidence map is not integrated into the 4DGS repo, we are trying to organize the simplified code in this repo.
If the generated images have sufficient consistency with some seed, feel free to ignore it.

## Acknowledgement

We have intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.

- [Syncdreamer](https://github.com/liuyuan-pal/SyncDreamer)
- [RIFE](https://github.com/hzwer/ECCV2022-RIFE)
- [Consistent4D](https://github.com/yanqinJiang/Consistent4D)
- [4DGS](https://github.com/fudan-zvg/4d-gaussian-splatting)


## BibTeX
If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:
```
@article{pan2024efficient4d,
  title={Efficient4D: Fast Dynamic 3D Object Generation from a Single-view Video},
  author={Pan, Zijie and Yang, Zeyu and Zhu, Xiatian and Zhang, Li},
  journal={arXiv preprint arXiv 2401.08742},
  year={2024}
}
```
