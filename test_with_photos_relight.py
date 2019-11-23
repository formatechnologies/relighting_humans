import sys
import os
from glob import glob
import time

import numpy as np
import cv2

import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as f

import model
import sh_rot
from SquarizeImage import SquarizeImage

from lighting.relighting_humans import RelightingHumans
relighting_humans_model = RelightingHumans()


indir_path = 'photo_inputs/'
outdir_path = 'photo_outputs/'

if not os.path.exists(outdir_path):
    os.makedirs(outdir_path)

img_paths = glob(indir_path + '*.jpg')

n_files = len(img_paths)

for i in range(n_files):
    file = img_paths[i]
    print('Processing [%03d/%03d] %s' % (i+1, n_files, file))

    img_orig = cv2.imread(file, cv2.IMREAD_COLOR)
    mask_orig = cv2.imread(file[:-4]+'_mask.png', cv2.IMREAD_GRAYSCALE)

    transport, albedo, light, shading, rendering = relighting_humans_model.run({'image': img_orig, 'seg_body': mask_orig})
    image_concat = (np.hstack((img_orig.astype(np.float32)/255, shading.clip(0, 1), albedo.clip(0, 1), rendering.clip(0, 1))) * 255).astype(np.uint8)

    basename = os.path.basename(file)[:-4]
    cv2.imwrite(outdir_path+os.path.basename(file), img_orig)
    cv2.imwrite(outdir_path+basename+'_mask.png', mask_orig)
    cv2.imwrite(outdir_path+basename+'_albedo.jpg', 255 * albedo)
    cv2.imwrite(outdir_path+basename+'_shading.jpg', 255 * shading)
    cv2.imwrite(outdir_path+basename+'_rendering.jpg', 255 * rendering)
    cv2.imwrite(outdir_path+basename+'_concat.jpg', image_concat)
    np.save(outdir_path+basename+'_light.npy', light)
    np.savez_compressed(outdir_path+basename+'_transport.npz', T=transport)







# parser = argparse.ArgumentParser(description='Relighting images')
# parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
# parser.add_argument('--shading', '-s', default=False, type=bool, help='Toggle shading animation')
# parser.add_argument('--in_dir', '-i', default='photo_outputs', help='Input directory')
# parser.add_argument('--out_dir', '-o', default='relighting_outputs', help='Output directory')
# parser.add_argument('--light_path', '-l', default='test_lights/cluster08.npy', help='Input directory')

# args = parser.parse_args()

# if not os.path.exists(args.out_dir):
#     os.makedirs(args.out_dir)

# outdir_path = args.out_dir if args.out_dir[:-1] == '/' else args.out_dir + '/'
# indir_path = args.in_dir if args.in_dir[:-1] == '/' else args.in_dir + '/'

# gpu = args.gpu
# xp = cuda.cupy if gpu > -1 else np
# if gpu>-1:
#     cuda.check_cuda_available()
#     cuda.get_device(gpu).use()

# def trim(img, mask, padding_x=5, padding_y=5):
#     mask_ids = np.where(mask>0)
#     y_max = min(max(mask_ids[0])+padding_y, img.shape[0])
#     y_min = max(min(mask_ids[0])-padding_y, 0)
#     x_max = min(max(mask_ids[1])+padding_x, img.shape[1])
#     x_min = max(min(mask_ids[1])-padding_x, 0)
#     if (y_max - y_min) % 2 == 1:
#         y_max -= 1
#     if (x_max - x_min) % 2 == 1:
#         x_max -= 1
#     return img[y_min:y_max,x_min:x_max]

# transport_postfix = '_transport.npz'
# transport_files = glob(indir_path + '*' + transport_postfix)
# transport_files.sort()

# n_files = len(transport_files)
# print('# files = %d' % n_files)

# light_basename = os.path.basename(args.light_path)[:-len('.npy')]
# print('Selected light: %s' % light_basename)

# light = np.load(args.light_path)
# if light.shape[0] < light.shape[1]:
#     light = light.T

# n_rotation_div = 72
# n_rotations = 10

# for i in range(n_files):
#     file = transport_files[i]
#     basepath = file[:-len(transport_postfix)]
#     basename = os.path.basename(file)[:-len(transport_postfix)]
#     print('Processing [%03d/%03d] %s' % (i+1, n_files, basename))

#     mask = cv2.imread(basepath + '_mask.png', cv2.IMREAD_COLOR)

#     albedo = cv2.imread(basepath + '_albedo.jpg', cv2.IMREAD_COLOR)
#     if albedo is None:
#         albedo = cv2.imread(basepath + '_albedo.png', cv2.IMREAD_COLOR)
#     transport = np.load(transport_files[i])['T']

#     if albedo is None:
#         print('Error: cannot open mask for "%s", skipping' % (basepath + '_albedo.jpg'))
#         continue

#     if transport is None:
#         print('Error: cannot open transport for "%s", skipping' % (basepath + '_transport.npz'))
#         continue

#     albedo = trim(albedo, mask)
#     transport = trim(transport, mask)

#     if gpu>-1:
#         albedo = cuda.to_gpu(albedo)
#         transport = cuda.to_gpu(transport)

#     tmp_renderings = []
#     max_val = 0.

#     for j in range(n_rotations * n_rotation_div):
#         deg = (360. / n_rotation_div) * j
#         R = sh_rot.calc_y_rot(deg / 180. * np.pi)
#         coeffs = np.empty_like(light)
#         coeffs[:,0] = sh_rot.sh_rot(R, light[:,0])
#         coeffs[:,1] = sh_rot.sh_rot(R, light[:,1])
#         coeffs[:,2] = sh_rot.sh_rot(R, light[:,2])

#         if gpu>-1:
#             coeffs = cuda.to_gpu(coeffs)

#         shading = xp.matmul(transport, coeffs)

#         if not args.shading:
#             rendering = albedo * shading
#         else:
#             rendering = shading

#         tmp_renderings.append(rendering)
#         max_val = max((max_val, xp.max(rendering)))

#     save_basepath = outdir_path + basename + '+' + light_basename
#     if not os.path.exists(save_basepath):
#         os.makedirs(save_basepath)

#     for j in range(n_rotations * n_rotation_div):
#         rendering = 255 * tmp_renderings[j] / max_val
#         if gpu>-1:
#             rendering = cuda.to_cpu(rendering)
#         cv2.imwrite(save_basepath + ('/frame%03d.jpg' % j), rendering)

#     video_path = save_basepath + '.mp4'
#     files_path = save_basepath + '/frame%03d.jpg'
#     os.system('ffmpeg -y -r 30 -i ' + files_path + ' -vcodec libx264 -pix_fmt yuv420p -r 60 ' + video_path)
