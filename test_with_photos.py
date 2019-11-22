from glob import glob
import numpy as np
import cv2
import os
import time
import sys
import model
import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F
from SquarizeImage import SquarizeImage
import argparse

parser = argparse.ArgumentParser(description='Relighting humans')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--in_dir', '-i', default='photo_inputs', help='Input directory')
parser.add_argument('--out_dir', '-o', default='photo_outputs', help='Output directory')
parser.add_argument('--model_path', '-p', default='models/model_060.chainer', help='Model path')

args = parser.parse_args()

indir_path = args.in_dir if args.in_dir[:-1] == '/' else args.in_dir + '/'
outdir_path = args.out_dir if args.out_dir[:-1] == '/' else args.out_dir + '/'

if not os.path.exists(outdir_path):
    os.makedirs(outdir_path)

img_paths = glob(indir_path + '*.jpg')

shared_model_file = args.model_path
gpu = args.gpu
target_image_size = 1024

m_shared = model.CNNAE2ResNet()
serializers.load_npz(shared_model_file, m_shared)
m_shared.train = True
m_shared.train_dropout = False

t_start = time.time()
sys.stdout.write('loading model data ... ')

sys.stdout.write('done (%.3f sec)\n' % (time.time() - t_start))

xp = cuda.cupy if gpu > -1 else np
if gpu>-1:
    cuda.check_cuda_available()
    cuda.get_device(gpu).use()
    m_shared.to_gpu()

def infer_light_transport_albedo_and_light(img, mask):
    img = 2.*img-1.
    img = img.transpose(2,0,1)
    mask3 = mask[None,:,:].repeat(3,axis=0).astype(np.float32)
    mask9 = mask[None,:,:].repeat(9,axis=0).astype(np.float32)

    if gpu>-1:
        img = cuda.to_gpu(img)
        mask3 = cuda.to_gpu(mask3)
        mask9 = cuda.to_gpu(mask9)

    img_batch = chainer.Variable(img[None,:,:,:], volatile=True)
    mask3_batch = chainer.Variable(mask3[None,:,:,:], volatile=True)
    mask9_batch = chainer.Variable(mask9[None,:,:,:], volatile=True)
    img_batch = mask3_batch * img_batch

    res_transport, res_albedo, res_light = m_shared(img_batch)

    res_transport = (mask9_batch * res_transport).data[0]
    res_albedo = (mask3_batch * res_albedo).data[0]
    res_light = res_light.data

    if gpu>-1:
        res_transport = cuda.to_cpu(res_transport)
        res_albedo = cuda.to_cpu(res_albedo)
        res_light = cuda.to_cpu(res_light)

    res_transport = res_transport.transpose(1,2,0)
    res_albedo = res_albedo.transpose(1,2,0)

    return res_transport, res_albedo, res_light

n_files = len(img_paths)

for i in range(n_files):
    file = img_paths[i]
    print('Processing [%03d/%03d] %s' % (i+1, n_files, file))
    
    img_orig = cv2.imread(file, cv2.IMREAD_COLOR)
    mask_orig = cv2.imread(file[:-4]+'_mask.png', cv2.IMREAD_GRAYSCALE)
    
    if img_orig is None:
        print('  Error: cannot open image: %s' % file)
        continue

    if mask_orig is None:
        print('  Error: cannot open mask: %s' % (file[:-4]+'_mask.png'))
        continue
    
    si = SquarizeImage(img_orig, mask_orig, 1024)
    
    if img_orig.shape[0] == target_image_size and img_orig.shape[1] == target_image_size:
        img = img_orig.copy()
        mask = mask_orig.copy()
    else:
        img = si.get_squared_image()
        mask = si.get_squared_mask()
    
    img = img.astype(np.float32) / 255.
    mask = mask.astype(np.float32) / 255.
    
    t_start = time.time()
    transport, albedo, light = infer_light_transport_albedo_and_light(img, mask)
    print('Inference time: %f sec' % (time.time() - t_start))

    basename = os.path.basename(file)[:-4]
    
    albedo = si.replace_image(np.zeros(img_orig.shape, dtype=albedo.dtype), mask_orig, albedo)
    transport = si.replace_image(np.zeros((img_orig.shape[0], img_orig.shape[1], 9), dtype=transport.dtype), mask_orig, transport)
    shading = np.matmul(transport, light)
    rendering = albedo * shading
    
    cv2.imwrite(outdir_path+os.path.basename(file), img_orig)
    cv2.imwrite(outdir_path+basename+'_albedo.jpg', 255 * albedo)

    cv2.imwrite(outdir_path+basename+'_mask.png', mask_orig)
    np.save(outdir_path+basename+'_light.npy', light)
    np.savez_compressed(outdir_path+basename+'_transport.npz', T=transport)

    cv2.imwrite(outdir_path+basename+'_shading.jpg', 255 * shading)
    cv2.imwrite(outdir_path+basename+'_rendering.jpg', 255 * rendering)
