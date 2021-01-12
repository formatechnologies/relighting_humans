import os
from glob import glob

import numpy as np
import cv2


from iris.lighting.relighting_humans import RelightingHumans
relighting_humans_model = RelightingHumans()

light_path = 'test_lights/cluster08.npy'
light_basename = os.path.basename(light_path)[:-len('.npy')]
print('Selected light: %s' % light_basename)
default_light = np.load(light_path)
if default_light.shape[0] < default_light.shape[1]:
    default_light = default_light.T

indir_path = 'photo_inputs/'
outdir_path = 'photo_outputs/'
relighting_path = 'relighting_outputs/'

if not os.path.exists(outdir_path):
    os.makedirs(outdir_path)
if not os.path.exists(relighting_path):
    os.makedirs(relighting_path)

img_paths = glob(indir_path + '*.jpg')
n_files = len(img_paths)
for i in range(n_files):
    file = img_paths[i]
    print('Processing [%03d/%03d] %s' % (i+1, n_files, file))

    img_orig = cv2.imread(file, cv2.IMREAD_COLOR)
    mask_orig = cv2.imread(file[:-4]+'_mask.png', cv2.IMREAD_GRAYSCALE)

    transport, albedo, light, shading, rendering = relighting_humans_model.run(img_orig, mask_orig)
    image_concat = np.hstack((img_orig.astype(np.float32)/255, shading.clip(0, 1), albedo.clip(0, 1), rendering.clip(0, 1)))
    renderings = relighting_humans_model.generate_renderings(img_orig, mask_orig, transport, albedo, default_light)

    basename = os.path.basename(file)[:-4]
    cv2.imwrite(outdir_path+os.path.basename(file), img_orig)
    cv2.imwrite(outdir_path+basename+'_mask.png', mask_orig)
    cv2.imwrite(outdir_path+basename+'_albedo.jpg', 255 * albedo)
    cv2.imwrite(outdir_path+basename+'_shading.jpg', 255 * shading)
    cv2.imwrite(outdir_path+basename+'_rendering.jpg', 255 * rendering)
    cv2.imwrite(outdir_path+basename+'_concat.jpg', 255 * image_concat)
    np.save(outdir_path+basename+'_light.npy', light)
    np.savez_compressed(outdir_path+basename+'_transport.npz', T=transport)

    save_basepath = relighting_path + basename + '+' + light_basename
    if not os.path.exists(save_basepath):
        os.makedirs(save_basepath)
    for j in range(len(renderings)):
        cv2.imwrite(save_basepath + ('/frame%03d.jpg' % j), 255 * renderings[j])
