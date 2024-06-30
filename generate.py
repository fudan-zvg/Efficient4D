import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave

from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config, prepare_inputs


def load_model(cfg, ckpt, strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=strict)
    model = model.cuda().eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/syncdreamer.yaml')
    parser.add_argument('--ckpt', type=str, default='ckpt/syncdreamer-pretrain.ckpt')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--elevation', type=float, required=True)

    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=-1)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--cfg_scales', type=float, nargs='+', default=[2.0, 1.0])
    parser.add_argument('--decomposed_sampling', action='store_true')
    parser.add_argument('--batch_view_num', type=int, default=16)
    parser.add_argument('--seed', type=int, default=6033)

    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_steps', type=int, default=50)

    parser.add_argument('--smooth_filter', action='store_true')
    parser.add_argument('--is_cyc', type=int, default=0)
    parser.add_argument('--frame_num', type=int, default=1)
    flags = parser.parse_args()

    torch.random.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    model = load_model(flags.cfg, flags.ckpt, strict=True)
    assert isinstance(model, SyncMultiviewDiffusion)
    Path(f'{flags.output}').mkdir(exist_ok=True, parents=True)

    # prepare data
    data = prepare_inputs(flags.input, flags.elevation, flags.crop_size, frame_num=flags.frame_num)
    for k, v in data.items():
        N = min(flags.frame_num, v.shape[0])
        print(f'total frames:{N}')
        data[k] = v[:N].cuda()
        # data[k] = torch.repeat_interleave(data[k], flags.sample_num, dim=0).cuda()

    if flags.sampler == 'ddim':
        sampler = SyncDDIMSampler(model, flags.sample_steps)
    else:
        raise NotImplementedError

    if flags.decomposed_sampling:
        mode = 'DG'
        print("Decomposed Guidance (DG) Sampling")
        print(f"unconditional scales {flags.cfg_scales}")
    else:
        mode = 'CFG'
        print("Classifier-Free Guidance (CFG) Sampling")
        print(f"unconditional scale {flags.cfg_scales[0]:.1f}")

    if flags.decomposed_sampling:
        x_sample = model.sample(sampler, data, flags.cfg_scales, flags.batch_view_num,
                                smooth_filter=flags.smooth_filter, is_cyc=flags.is_cyc)
    else:
        x_sample = model.sample(sampler, data, flags.cfg_scales[0], flags.batch_view_num,
                                smooth_filter=flags.smooth_filter, is_cyc=flags.is_cyc)

    B, N, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample, max=1.0, min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    for bi in range(B):
        output_fn = Path(flags.output) / f'{bi}.png'
        imsave(output_fn, np.concatenate([x_sample[bi, ni] for ni in range(N)], 1))


if __name__ == "__main__":
    main()
