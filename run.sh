NAME=$1
OUTNAME=$2
DEVICE=$3
N_FRAME=$4
cyc=$5

INPUT="./data/$NAME"
OUTPUT="./output/$OUTNAME"
OUTPUT_RECON="./recon_data/$OUTNAME/"
echo $OUTPUT

# generate pseudo muti-view and muti-frame consistent images
CUDA_VISIBLE_DEVICES=$DEVICE python generate.py \
  --input $INPUT \
  --output $OUTPUT \
  --crop_size 200 \
  --elevation 0 \
  --frame_num $N_FRAME \
  --smooth_filter \
  --decomposed_sampling \
  --is_cyc $cyc \
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


