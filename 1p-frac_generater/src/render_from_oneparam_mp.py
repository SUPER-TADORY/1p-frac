import os
import string
import shutil
import glob
import random
import argparse
import numpy as np
from multiprocessing import Pool
from multiprocessing import current_process

from ifs_function import ifs_function
import utils


def reshape(params):
    a = params[:,0,0]
    b = params[:,0,1]
    c = params[:,1,0]
    d = params[:,1,1]
    e = params[:,0,2]
    f = params[:,1,2]
    
    return np.stack([a,b,c,d,e,f]).T

def calc_p(params):
    tmp_l = []
    As = params[:,:,:2]
    for A in As:
        a = abs(np.linalg.det(A))
        tmp_l.append(a)
    tmp_l = np.array(tmp_l)
    p = (tmp_l / tmp_l.sum()).reshape(-1, 1)
    flatten_params = reshape(params)
    params = np.concatenate([flatten_params, p], -1)

    return params


def generate_image(img_savedir, shapename, params, args, sample):
    while 1:
        name = '%02d' % sample

        padded_fractal_weight = name
        cls_savedir = f'{img_savedir}/{name}'
        tmp_l = [o for o in glob.glob(f'{cls_savedir}/*.png')]
        if len(tmp_l) > 0:
            break

        os.makedirs(cls_savedir, exist_ok=True)
        generators = ifs_function(prev_x=0.0, prev_y=0.0, save_root=img_savedir, fractal_name=name, fractal_weight_count=padded_fractal_weight, mask_mode=args.mask_mode)

        if len(params.shape) == 3:
            params = calc_p(params)

        for param in params:
            # Add perturbation (=args.delta)
            noise_l = [random.uniform(1 - args.delta, 1 + args.delta) for _ in range(6)]
            generators.set_param(float(param[0]), float(param[1]), float(param[2]), float(param[3]), float(param[4]), float(param[5]), float(param[6]),
                weight_a=noise_l[0], weight_b=noise_l[1], weight_c = noise_l[2], weight_d=noise_l[3], weight_e=noise_l[4] , weight_f=noise_l[5] )
        generators.calculate(args.iteration)
        result = generators.draw_patch(args.image_size_x, args.image_size_y, args.pad_size_x, args.pad_size_y, 'gray', 0, threshold = args.pixel_threshold)
        # print(result)
        if result:
            print(shapename)
            break
        else:
            shutil.rmtree(cls_savedir)
            continue


def generate_image_wrapper(args_tuple):
    return generate_image(*args_tuple)


def generate_randomname(length=8):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for _ in range(length))


def set_baseparams(args, margin=0.):
    n_range = [args.sigma - margin, args.sigma + margin]
    base_params, sigma = utils.sample_system(args.N, beta=n_range)

    return base_params, sigma


def main(args):
    random_id = generate_randomname(length=8)
    savename = f'sigma{args.sigma}_perturb{args.delta}_sample{args.sample}_{random_id}'
    img_savedir = f'{args.img_basesavedir}/{savename}'
    os.makedirs(img_savedir, exist_ok=True)

    # Set single fratal based on sigma-factor
    base_params, sigma = set_baseparams(args, margin=0.1)

    tasks = []

    for sample in range(args.sample):
        tasks.append((img_savedir, savename, base_params, args, sample))

    # Generation
    with Pool() as pool:
        pool.map(generate_image_wrapper, tasks, chunksize=1)

    print(f"Total images generated or confirmed existing: {args.sample}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_basesavedir', type=str, required=True)
    parser.add_argument('--mask_mode', type=str, default='random', choices=['random', 'fill'])
    parser.add_argument('--sample', type=int, default=1000, help='sample num from delta distribution')
    parser.add_argument('--image_size_x', default=512, type = int, help='image size x')
    parser.add_argument('--image_size_y', default=512, type = int, help='image size y')
    parser.add_argument('--pad_size_x', default=6, type = int, help='padding size x')
    parser.add_argument('--pad_size_y', default=6, type = int, help='padding size y')
    parser.add_argument('--iteration', default=100000, type = int, help='iteration')
    parser.add_argument('--draw_type', default='patch_gray', type = str, help='{point, patch}_{gray, color}')
    parser.add_argument('--pixel_threshold', type=float, default=0.05, help='Filling rate')
    parser.add_argument('--N', type=int, default=2, help='Affine num for single fractal')
    parser.add_argument('--sigma', type=float, default=3.5, help='Sigma factor')
    parser.add_argument('--delta', type=float, default=0.1, help='Perturbation strength to fractal parameters')
    args = parser.parse_args()
    main(args)
