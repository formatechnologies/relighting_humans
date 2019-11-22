#!/usr/bin/env python
# coding: utf-8

from glob import glob
import numpy as np
np.random.seed(1234)
import skimage.morphology
import cv2
import os
import sys
import random
import time
import model
import chainer
from chainer import cuda
from chainer import serializers
from chainer.functions.loss.mean_absolute_error import mean_absolute_error
import chainer.functions as F
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Relighting humans')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value means CPU)')
parser.add_argument('--max_epochs', default=10000, type=int, help='Max number of epochs')
parser.add_argument('--train_dir', '-i', default='train_inputs', help='Directory for training input images')
parser.add_argument('--test_dir', '-t', default='test_inputs', help='Directory for test input images')
parser.add_argument('--out_dir', '-o', default='outputs', help='Directory for output images')
parser.add_argument('--train_light_dir', '-l0', default='train_lights', help='Light directory for training')
parser.add_argument('--test_light_dir', '-l1', default='test_lights', help='Light directory for test')
parser.add_argument('--w_transport', '-tw0', default=1., type=float, help='')
parser.add_argument('--w_albedo', '-tw1', default=1., type=float, help='')
parser.add_argument('--w_light', '-tw2', default=1., type=float, help='')
parser.add_argument('--w_transport_tv', '-tw3', default=1., type=float, help='')
parser.add_argument('--w_albedo_tv', '-tw4', default=1., type=float, help='')
parser.add_argument('--w_shading_transport', '-tw5', default=1., type=float, help='')
parser.add_argument('--w_shading_light', '-tw6', default=1., type=float, help='')
parser.add_argument('--w_shading_all', '-tw7', default=1., type=float, help='')
parser.add_argument('--w_rendering_albedo', '-tw8', default=1., type=float, help='')
parser.add_argument('--w_rendering_transport', '-tw9', default=1., type=float, help='')
parser.add_argument('--w_rendering_light', '-tw10', default=1., type=float, help='')
parser.add_argument('--w_rendering_albedo_transport', '-tw11', default=1., type=float, help='')
parser.add_argument('--w_rendering_transport_light', '-tw12', default=1., type=float, help='')
parser.add_argument('--w_rendering_albedo_light', '-tw13', default=1., type=float, help='')
parser.add_argument('--w_rendering_all', '-tw14', default=1., type=float, help='')

args = parser.parse_args()

traindir_path = args.train_dir if args.train_dir[:-1] == '/' else args.train_dir + '/'
testdir_path = args.test_dir if args.test_dir[:-1] == '/' else args.test_dir + '/'
outdir_path = args.out_dir if args.out_dir[:-1] == '/' else args.out_dir + '/'
train_light_path = args.train_light_dir if args.train_light_dir[:-1] == '/' else args.train_light_dir + '/'
test_light_path = args.test_light_dir if args.test_light_dir[:-1] == '/' else args.test_light_dir + '/'

if not os.path.exists(outdir_path):
    os.makedirs(outdir_path)

model_save_path = outdir_path + 'shared_model_%03d.chainer'
opt_save_path = outdir_path + 'shared_opt_%03d.chainer'

gpu = args.gpu
max_epoch = args.max_epochs
batch_size = 1

w_transport = args.w_transport
w_albedo = args.w_albedo
w_light = args.w_light
w_transport_tv = args.w_transport_tv
w_albedo_tv = args.w_albedo_tv
w_shading_transport = args.w_shading_transport
w_shading_light = args.w_shading_light
w_shading_all = args.w_shading_all
w_rendering_albedo = args.w_rendering_albedo
w_rendering_transport = args.w_rendering_transport
w_rendering_light = args.w_rendering_light
w_rendering_albedo_transport = args.w_rendering_albedo_transport
w_rendering_transport_light = args.w_rendering_transport_light
w_rendering_albedo_light = args.w_rendering_albedo_light
w_rendering_all = args.w_rendering_all

train_fpath  = glob(traindir_path+"*_tex.png")
train_light_fpath = glob(train_light_path+'*.npy')
test_fpath = glob(testdir_path+"*_tex.png")
test_light_fpath = glob(test_light_path+'*.npy')

if len(train_fpath) == 0:
    print('Error: no training images')
    exit(-1)

if len(train_light_fpath) == 0:
    print('Error: no training lights')
    exit(-1)

if len(test_fpath) == 0:
    print('Error: no test images')
    exit(-1)

if len(test_light_fpath) == 0:
    print('Error: no test lights')
    exit(-1)

train_light_fpath = train_light_fpath[:1]
test_light_fpath = test_light_fpath[:1]

m_shared = model.CNNAE2ResNet()

o_shared = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
o_shared.setup(m_shared)

def save_shading_image(light_transport_map, light, filename):
    cv2.imwrite(filename, 255 * np.matmul(light_transport_map, light))

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

xp = cuda.cupy if gpu > -1 else np
if gpu>-1:
    cuda.check_cuda_available()
    cuda.get_device(gpu).use()
    m_shared.to_gpu()

N_train_img = len(train_fpath)
N_train_light = len(train_light_fpath)
N_train_total = N_train_img * N_train_light

N_test_img = len(test_fpath)
N_test_light = len(test_light_fpath)
N_test_total = N_test_img * N_test_light

print('Preloading %d lights ...' % (N_train_light + N_test_light))
train_lights = []
for train_light_path in train_light_fpath:
    light = np.load(train_light_path)
    if gpu>-1:
        light = cuda.to_gpu(light)
    train_lights.append(light)
test_lights = []
for train_light_path in test_light_fpath:
    light = np.load(train_light_path)
    if gpu>-1:
        light = cuda.to_gpu(light)
    test_lights.append(light)

v_kernel = np.zeros((9,9,3,3),dtype=np.float32)
h_kernel = v_kernel.copy()
v_kernel[np.identity(9).astype(np.bool)] = np.array([[0,1,0],[0,-1,0],[0,0,0]], dtype=np.float32)
h_kernel[np.identity(9).astype(np.bool)] = np.array([[0,0,0],[0,-1,1],[0,0,0]], dtype=np.float32)
if gpu>-1:
    v_kernel = cuda.to_gpu(v_kernel)
    h_kernel = cuda.to_gpu(h_kernel)
    
v_kernel3 = np.zeros((3,3,3,3),dtype=np.float32)
h_kernel3 = v_kernel3.copy()
v_kernel3[np.identity(3).astype(np.bool)] = np.array([[0,1,0],[0,-1,0],[0,0,0]], dtype=np.float32)
h_kernel3[np.identity(3).astype(np.bool)] = np.array([[0,0,0],[0,-1,1],[0,0,0]], dtype=np.float32)

if gpu>-1:
    v_kernel3 = cuda.to_gpu(v_kernel3)
    h_kernel3 = cuda.to_gpu(h_kernel3)

for epoch in range(max_epoch):
    print('[%s] epoch: %d, gpu: %d\n  outdir: %s' % (sys.argv[0], epoch, gpu, outdir_path))
    start_epoch = time.time()
    
    L_sum = 0.
    L_transport_sum = 0.
    L_albedo_sum = 0.
    L_light_sum = 0.
    L_transport_tv_sum = 0.
    L_albedo_tv_sum = 0.
    L_shading_transport_sum = 0.
    L_shading_light_sum = 0.
    L_shading_all_sum = 0.
    L_rendering_albedo_sum = 0.
    L_rendering_transport_sum = 0.
    L_rendering_light_sum = 0.
    L_rendering_albedo_transport_sum = 0.
    L_rendering_transport_light_sum = 0.
    L_rendering_albedo_light_sum = 0.
    L_rendering_all_sum = 0.
    perm_img = np.random.permutation(N_train_img)
    perm_light = np.random.permutation(N_train_light)
    
    pbar = tqdm(total=N_train_total, desc='  train', ascii=True)
    for bi in range(N_train_img):
        for i in perm_img[bi:bi+1]:
            albedo = cv2.imread(train_fpath[i], cv2.IMREAD_COLOR).astype(np.float32) / 255.
            mask = cv2.imread(train_fpath[i][:-7]+"mask.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            eroded_mask = skimage.morphology.binary_erosion(mask).astype(np.float32)
            transport = np.load(train_fpath[i][:-7]+"transport.npz")['T']

            if gpu>-1:
                albedo = cuda.to_gpu(albedo)
                mask = cuda.to_gpu(mask)
                eroded_mask = cuda.to_gpu(eroded_mask)
                transport = cuda.to_gpu(transport)
            
            albedo_batch = chainer.Variable(albedo.transpose(2,0,1)[None,:,:])
            erode3_batch = chainer.Variable(eroded_mask[None,:,:].repeat(3,axis=0)[None,:,:,:])
            erode9_batch = chainer.Variable(eroded_mask[None,:,:].repeat(9,axis=0)[None,:,:,:])
            mask3_batch = chainer.Variable(mask[None,:,:].repeat(3,axis=0)[None,:,:,:])
            mask9_batch = chainer.Variable(mask[None,:,:].repeat(9,axis=0)[None,:,:,:])
            transport_batch = chainer.Variable(transport.transpose(2,0,1)[None,:,:,:])

            transport_reshaped_batch = F.transpose(transport_batch, axes=(0,2,3,1))
            transport_reshaped_batch = F.reshape(transport_reshaped_batch, (-1, 9))

            zero_transport = xp.zeros_like(transport_batch.data)
            zero_albedo = xp.zeros_like(albedo_batch.data)

            for bj in range(N_train_light):
                for j in perm_light[bj:bj+1]:
                    light = train_lights[j]
                    light_batch = chainer.Variable(light)

                    shading = xp.matmul(transport, light)
                    shading = xp.clip(shading, 0., 10.)
                    shading_batch = chainer.Variable(shading.transpose(2,0,1)[None,:,:,:])

                    rendering = albedo * shading
                    rendering_batch = chainer.Variable(rendering.transpose(2,0,1)[None,:,:,:])

                    img = 2.*rendering-1.
                    img_batch = chainer.Variable(img.transpose(2,0,1)[None,:,:,:])
                    img_batch = mask3_batch * img_batch

                    transport_hat, albedo_hat, light_hat = m_shared(img_batch)
                    
                    transport_hat = mask9_batch * transport_hat
                    L_transport = mean_absolute_error(transport_hat, transport_batch)
                    
                    transport_hat_dy = erode9_batch * F.convolution_2d(transport_hat, W=v_kernel, pad=1)
                    transport_hat_dx = erode9_batch * F.convolution_2d(transport_hat, W=h_kernel, pad=1)
                    L_transport_tv = (mean_absolute_error(transport_hat_dy, zero_transport) + mean_absolute_error(transport_hat_dx, zero_transport))
                    
                    albedo_hat = mask3_batch * albedo_hat
                    L_albedo = mean_absolute_error(albedo_hat, albedo_batch)
                    
                    albedo_hat_dy = erode3_batch * F.convolution_2d(albedo_hat, W=v_kernel3, pad=1)
                    albedo_hat_dx = erode3_batch * F.convolution_2d(albedo_hat, W=h_kernel3, pad=1)
                    L_albedo_tv = (mean_absolute_error(albedo_hat_dy, zero_albedo) + mean_absolute_error(albedo_hat_dx, zero_albedo))

                    L_light = mean_absolute_error(F.expand_dims(light_hat, axis=0), F.expand_dims(light_batch, axis=0))

                    transport_reshaped_hat = F.transpose(transport_hat, axes=(0,2,3,1))
                    transport_reshaped_hat = F.reshape(transport_reshaped_hat, (-1, 9))
                    shading_transport_hat = F.matmul(transport_reshaped_hat, light_batch)
                    shading_transport_hat = F.clip(shading_transport_hat, 0., 10.)
                    shading_transport_hat = F.reshape(shading_transport_hat, (1, albedo.shape[0], albedo.shape[1], 3))
                    shading_transport_hat = F.transpose(shading_transport_hat, axes=(0,3,1,2))
                    L_shading_transport = mean_absolute_error(shading_transport_hat, shading_batch)

                    shading_light_hat = F.matmul(transport_reshaped_batch, light_hat)
                    shading_light_hat = F.clip(shading_light_hat, 0., 10.)
                    shading_light_hat = F.reshape(shading_light_hat, (1, albedo.shape[0], albedo.shape[1], 3))
                    shading_light_hat = F.transpose(shading_light_hat, axes=(0,3,1,2))
                    L_shading_light = mean_absolute_error(shading_light_hat, shading_batch)

                    shading_all_hat = F.matmul(transport_reshaped_hat, light_hat)
                    shading_all_hat = F.clip(shading_all_hat, 0., 10.)
                    shading_all_hat = F.reshape(shading_all_hat, (1, albedo.shape[0], albedo.shape[1], 3))
                    shading_all_hat = F.transpose(shading_all_hat, axes=(0,3,1,2))
                    L_shading_all = mean_absolute_error(shading_all_hat, shading_batch)

                    rendering_albedo_hat = albedo_hat * shading_batch
                    L_rendering_albedo = mean_absolute_error(rendering_albedo_hat, rendering_batch)
                    
                    rendering_transport_hat = albedo_batch * shading_transport_hat
                    L_rendering_transport = mean_absolute_error(rendering_transport_hat, rendering_batch)
                    
                    rendering_light_hat = albedo_batch * shading_light_hat
                    L_rendering_light = mean_absolute_error(rendering_light_hat, rendering_batch)
                    
                    rendering_albedo_transport_hat = albedo_hat * shading_transport_hat
                    L_rendering_albedo_transport = mean_absolute_error(rendering_albedo_transport_hat, rendering_batch)
                    
                    rendering_transport_light_hat = albedo_batch * shading_light_hat
                    L_rendering_transport_light = mean_absolute_error(rendering_transport_light_hat, rendering_batch)

                    rendering_albedo_light_hat = albedo_hat * shading_light_hat
                    L_rendering_albedo_light = mean_absolute_error(rendering_albedo_light_hat, rendering_batch)
                    
                    rendering_all_hat = albedo_hat * shading_all_hat
                    L_rendering_all = mean_absolute_error(rendering_all_hat, rendering_batch)
                    
                    L = w_transport * L_transport + w_transport_tv * L_transport_tv + w_albedo * L_albedo + \
                        w_albedo_tv * L_albedo_tv + w_light * L_light + w_shading_transport * L_shading_transport + \
                        w_shading_light * L_shading_light + w_shading_all * L_shading_all + w_rendering_albedo * L_rendering_albedo + \
                        w_rendering_transport * L_rendering_transport + w_rendering_light * L_rendering_light + \
                        w_rendering_albedo_transport * L_rendering_albedo_transport + w_rendering_transport_light * L_rendering_transport_light + \
                        w_rendering_albedo_light * L_rendering_albedo_light + w_rendering_all * L_rendering_all

                    m_shared.cleargrads()
                    L.backward()
                    o_shared.update()
                    
                    L_sum += L_transport.data + L_albedo.data + L_light.data + L_transport_tv.data + L_albedo_tv.data + \
                        L_shading_transport.data + L_shading_light.data + L_shading_all.data + L_rendering_albedo.data + \
                        L_rendering_transport.data + L_rendering_light.data + L_rendering_albedo_transport.data + \
                        L_rendering_transport_light.data + L_rendering_albedo_light.data + L_rendering_all.data
                    L_transport_sum += L_transport.data
                    L_albedo_sum += L_albedo.data
                    L_light_sum += L_light.data
                    L_transport_tv_sum += L_transport_tv.data
                    L_albedo_tv_sum += L_albedo_tv.data
                    L_shading_transport_sum += L_shading_transport.data
                    L_shading_light_sum += L_shading_light.data
                    L_shading_all_sum += L_shading_all.data
                    L_rendering_albedo_sum += L_rendering_albedo.data
                    L_rendering_transport_sum += L_rendering_transport.data
                    L_rendering_light_sum += L_rendering_light.data
                    L_rendering_albedo_transport_sum += L_rendering_albedo_transport.data
                    L_rendering_transport_light_sum += L_rendering_transport_light.data
                    L_rendering_albedo_light_sum += L_rendering_albedo_light.data
                    L_rendering_all_sum += L_rendering_all.data

                    pbar.update(1)

    pbar.close()
    time_epoch = time.time() - start_epoch
    raw_L = L_sum / N_train_total
    raw_L_transport = L_transport_sum / N_train_total
    raw_L_albedo = L_albedo_sum / N_train_total
    raw_L_light = L_light_sum / N_train_total
    if gpu>-1:
        raw_L = cuda.to_cpu(raw_L)
        raw_L_transport = cuda.to_cpu(raw_L_transport)
        raw_L_albedo = cuda.to_cpu(raw_L_albedo)
        raw_L_light = cuda.to_cpu(raw_L_light)

    print('    loss[total/albedo/transport/light] = [%f,%f,%f,%f], time = %.3f [sec]' % 
        (raw_L, raw_L_albedo, raw_L_transport, raw_L_light, time_epoch))
    
    L_sum = 0.
    L_transport_sum = 0.
    L_albedo_sum = 0.
    L_light_sum = 0.
    L_transport_tv_sum = 0.
    L_albedo_tv_sum = 0.
    L_shading_transport_sum = 0.
    L_shading_light_sum = 0.
    L_shading_all_sum = 0.
    L_rendering_albedo_sum = 0.
    L_rendering_transport_sum = 0.
    L_rendering_light_sum = 0.
    L_rendering_albedo_transport_sum = 0.
    L_rendering_transport_light_sum = 0.
    L_rendering_albedo_light_sum = 0.
    L_rendering_all_sum = 0.    
    perm_test_img = np.random.permutation(N_test_img)
    perm_test_light = np.random.permutation(N_test_light)

    start_test = time.time()

    m_shared.train_dropout = False
    pbar = tqdm(total=N_test_total, desc='  test', ascii=True)
    for bi in range(N_test_img):
        for i in perm_test_img[bi:bi+1]:
            albedo = cv2.imread(test_fpath[i], cv2.IMREAD_COLOR).astype(np.float32) / 255.
            mask = cv2.imread(test_fpath[i][:-7]+"mask.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            eroded_mask = skimage.morphology.binary_erosion(mask).astype(np.float32)
            transport = np.load(test_fpath[i][:-7]+"transport.npz")['T']
            
            if gpu>-1:
                albedo = cuda.to_gpu(albedo)
                mask = cuda.to_gpu(mask)
                eroded_mask = cuda.to_gpu(eroded_mask)
                transport = cuda.to_gpu(transport)
            
            albedo_batch = chainer.Variable(albedo.transpose(2,0,1)[None,:,:,:], volatile=True)
            erode3_batch = chainer.Variable(eroded_mask[None,:,:].repeat(3,axis=0)[None,:,:,:], volatile=True)
            erode9_batch = chainer.Variable(eroded_mask[None,:,:].repeat(9,axis=0)[None,:,:,:], volatile=True)
            mask3_batch = chainer.Variable(mask[None,:,:].repeat(3,axis=0)[None,:,:,:], volatile=True)
            mask9_batch = chainer.Variable(mask[None,:,:].repeat(9,axis=0)[None,:,:,:], volatile=True)
            transport_batch = chainer.Variable(transport.transpose(2,0,1)[None,:,:,:], volatile=True)

            transport_reshaped_batch = F.transpose(transport_batch, axes=(0,2,3,1))
            transport_reshaped_batch = F.reshape(transport_reshaped_batch, (-1, 9))

            transport_dy = F.convolution_2d(transport_batch, W=v_kernel, pad=1)
            transport_dx = F.convolution_2d(transport_batch, W=h_kernel, pad=1)

            albedo_dy = F.convolution_2d(albedo_batch, W=v_kernel3, pad=1)
            albedo_dx = F.convolution_2d(albedo_batch, W=h_kernel3, pad=1)

            for bj in range(N_test_light):
                for j in perm_test_light[bj:bj+1]:
                    light = test_lights[j]
                    light_batch = chainer.Variable(xp.array(light), volatile=True)
 
                    shading = xp.matmul(transport, light)
                    shading = xp.clip(shading, 0., 10.)
                    shading_batch = chainer.Variable(shading.transpose(2,0,1)[None,:,:,:], volatile=True)
 
                    rendering = albedo * shading
                    rendering_batch = chainer.Variable(rendering.transpose(2,0,1)[None,:,:,:], volatile=True)
 
                    img = 2.*rendering-1.
                    img_batch = chainer.Variable(img.transpose(2,0,1)[None,:,:,:], volatile=True)
                    img_batch = mask3_batch * img_batch
 
                    transport_hat, albedo_hat, light_hat = m_shared(img_batch)
                    
                    transport_hat = mask9_batch * transport_hat
                    L_transport = mean_absolute_error(transport_hat, transport_batch)
                     
                    transport_hat_dy = F.convolution_2d(transport_hat, W=v_kernel, pad=1)
                    transport_hat_dx = F.convolution_2d(transport_hat, W=h_kernel, pad=1)
                    L_transport_tv = mean_absolute_error(transport_hat_dy, transport_dy) + mean_absolute_error(transport_hat_dx, transport_dx)

                    albedo_hat = mask3_batch * albedo_hat
                    L_albedo = mean_absolute_error(albedo_hat, albedo_batch)

                    zero_albedo = xp.zeros_like(albedo_batch.data)
                    albedo_hat_dy = F.convolution_2d(albedo_hat, W=v_kernel3, pad=1)
                    albedo_hat_dx = F.convolution_2d(albedo_hat, W=h_kernel3, pad=1)
                    L_albedo_tv = mean_absolute_error(albedo_hat_dy, albedo_dy) + mean_absolute_error(albedo_hat_dx, albedo_dx)

                    L_light = mean_absolute_error(F.expand_dims(light_hat, axis=0), F.expand_dims(light_batch, axis=0))
                    
                    transport_reshaped_hat = F.transpose(transport_hat, axes=(0,2,3,1))
                    transport_reshaped_hat = F.reshape(transport_reshaped_hat, (-1, 9))
                    shading_transport_hat = F.matmul(transport_reshaped_hat, light_batch)
                    shading_transport_hat = F.clip(shading_transport_hat, 0., 10.)
                    shading_transport_hat = F.reshape(shading_transport_hat, (1, albedo.shape[0], albedo.shape[1], 3))
                    shading_transport_hat = F.transpose(shading_transport_hat, axes=(0,3,1,2))
                    L_shading_transport = mean_absolute_error(shading_transport_hat, shading_batch)

                    shading_light_hat = F.matmul(transport_reshaped_batch, light_hat)
                    shading_light_hat = F.clip(shading_light_hat, 0., 10.)
                    shading_light_hat = F.reshape(shading_light_hat, (1, albedo.shape[0], albedo.shape[1], 3))
                    shading_light_hat = F.transpose(shading_light_hat, axes=(0,3,1,2))
                    L_shading_light = mean_absolute_error(shading_light_hat, shading_batch)

                    shading_all_hat = F.matmul(transport_reshaped_hat, light_hat)
                    shading_all_hat = F.clip(shading_all_hat, 0., 10.)
                    shading_all_hat = F.reshape(shading_all_hat, (1, albedo.shape[0], albedo.shape[1], 3))
                    shading_all_hat = F.transpose(shading_all_hat, axes=(0,3,1,2))
                    L_shading_all = mean_absolute_error(shading_all_hat, shading_batch)

                    rendering_albedo_hat = albedo_hat * shading_batch
                    L_rendering_albedo = mean_absolute_error(rendering_albedo_hat, rendering_batch)
                    
                    rendering_transport_hat = albedo_batch * shading_transport_hat
                    L_rendering_transport = mean_absolute_error(rendering_transport_hat, rendering_batch)
                    
                    rendering_light_hat = albedo_batch * shading_light_hat
                    L_rendering_light = mean_absolute_error(rendering_light_hat, rendering_batch)
                    
                    rendering_albedo_transport_hat = albedo_hat * shading_transport_hat
                    L_rendering_albedo_transport = mean_absolute_error(rendering_albedo_transport_hat, rendering_batch)
                    
                    rendering_transport_light_hat = albedo_batch * shading_light_hat
                    L_rendering_transport_light = mean_absolute_error(rendering_transport_light_hat, rendering_batch)

                    rendering_albedo_light_hat = albedo_hat * shading_light_hat
                    L_rendering_albedo_light = mean_absolute_error(rendering_albedo_light_hat, rendering_batch)
                    
                    rendering_all_hat = albedo_hat * shading_all_hat
                    L_rendering_all = mean_absolute_error(rendering_all_hat, rendering_batch)
                    
                    L_sum += L_transport.data + L_albedo.data + L_light.data + L_transport_tv.data + L_albedo_tv.data + \
                        L_shading_transport.data + L_shading_light.data + L_shading_all.data + L_rendering_albedo.data + \
                        L_rendering_transport.data + L_rendering_light.data + L_rendering_albedo_transport.data + \
                        L_rendering_transport_light.data + L_rendering_albedo_light.data + L_rendering_all.data
                    L_transport_sum += L_transport.data
                    L_albedo_sum += L_albedo.data
                    L_light_sum += L_light.data
                    L_transport_tv_sum += L_transport_tv.data
                    L_albedo_tv_sum += L_albedo_tv.data
                    L_shading_transport_sum += L_shading_transport.data
                    L_shading_light_sum += L_shading_light.data
                    L_shading_all_sum += L_shading_all.data
                    L_rendering_albedo_sum += L_rendering_albedo.data
                    L_rendering_transport_sum += L_rendering_transport.data
                    L_rendering_light_sum += L_rendering_light.data
                    L_rendering_albedo_transport_sum += L_rendering_albedo_transport.data
                    L_rendering_transport_light_sum += L_rendering_transport_light.data
                    L_rendering_albedo_light_sum += L_rendering_albedo_light.data
                    L_rendering_all_sum += L_rendering_all.data

                    pbar.update(1)
    pbar.close()
    m_shared.train_dropout = True

    time_test = time.time() - start_test
    raw_L = L_sum / N_test_total
    raw_L_transport = L_transport_sum / N_test_total
    raw_L_albedo = L_albedo_sum / N_test_total
    raw_L_light = L_light_sum / N_test_total
    if gpu>-1:
        raw_L = cuda.to_cpu(raw_L)
        raw_L_transport = cuda.to_cpu(raw_L_transport)
        raw_L_albedo = cuda.to_cpu(raw_L_albedo)
        raw_L_light = cuda.to_cpu(raw_L_light)
    print('    loss[total/albedo/transport/light] = [%f,%f,%f,%f], time = %.3f [sec]' % 
        (raw_L, raw_L_albedo, raw_L_transport, raw_L_light, time_test))

    # train
    
    albedo = cv2.imread(train_fpath[perm_img[0]], cv2.IMREAD_COLOR).astype(np.float32) / 255.
    mask = cv2.imread(train_fpath[perm_img[0]][:-7]+"mask.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    transport = np.load(train_fpath[perm_img[0]][:-7]+"transport.npz")['T']
    light = np.load(train_light_fpath[perm_light[0]])
    img = albedo * np.matmul(transport, light)
    res_transport, res_albedo, res_light = infer_light_transport_albedo_and_light(img, mask)
    cv2.imwrite(outdir_path+("train_albedo_epoch%03d.png" % epoch), 255 * res_albedo)
    np.savez_compressed(outdir_path+"train_transport_epoch%03d.npz" % epoch, T=res_transport)
    save_shading_image(res_transport, light, outdir_path+("train_shading_epoch%03d.png" % epoch))
    np.save(outdir_path+("train_light_epoch%03d.npy" % epoch), res_light)

    # test
    
    albedo = cv2.imread(test_fpath[perm_test_img[0]], cv2.IMREAD_COLOR).astype(np.float32) / 255.
    mask = cv2.imread(test_fpath[perm_test_img[0]][:-7]+"mask.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    transport = np.load(test_fpath[perm_test_img[0]][:-7]+"transport.npz")['T']
    light = np.load(test_light_fpath[perm_test_light[0]])
    img = albedo * np.matmul(transport, light)
    res_transport, res_albedo, res_light = infer_light_transport_albedo_and_light(img, mask)
    cv2.imwrite(outdir_path+("test_albedo_epoch%03d.png" % epoch), 255 * res_albedo)
    np.savez_compressed(outdir_path+"test_transport_epoch%03d.npz" % epoch, T=res_transport)
    save_shading_image(res_transport, light, outdir_path+("test_shading_epoch%03d.png" % epoch))
    np.save(outdir_path+("test_light_epoch%03d.npy" % epoch), res_light)

    if epoch > 0 and epoch % 2 == 0:
        serializers.save_npz(model_save_path % epoch, m_shared)
        serializers.save_npz(opt_save_path % epoch, o_shared)
