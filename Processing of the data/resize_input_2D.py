import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import re

dataset = "SKB"

# # CHECK THE SIZES
# train_images = ['imagesTr/' + img_name for img_name in sorted(os.listdir(dataset + '/imagesTr'))]
# test_images = ['imagesTs/' + img_name for img_name in sorted(os.listdir(dataset + '/imagesTs'))]
# train_labels = ['labelsTr/' + img_name for img_name in sorted(os.listdir(dataset + '/labelsTr'))]
# test_labels = ['labelsTs/' + img_name for img_name in sorted(os.listdir(dataset + '/labelsTs'))]
# imgs = train_images + test_images + train_labels + test_labels

# for img_name in imgs:
#   # load label
#   img_path = os.path.join(dataset, img_name)
#   img = nib.load(img_path).get_fdata() # 3D numpy matrix with an image

#   print(img_name, img.shape)




# CHANGE THE SIZES

masks_wrong_size = sorted(os.listdir(dataset + '/imagesTs_wrong_size'))
target_dir = "imagesTs"

count = 0

for mask_name in masks_wrong_size:
  # load label
  mask_path = os.path.join(dataset, "imagesTs_wrong_size", mask_name)
  old_nifti = nib.load(mask_path)
  old_mask = old_nifti.get_fdata() # 3D numpy matrix with an image
  affinity = old_nifti.affine

  # if the images have only 2 dimensions
  if len(old_mask.shape) == 2:
    count += 1
    new_shape = (old_mask.shape[0], old_mask.shape[1], 1)
    new_mask = np.zeros(new_shape)

    for row in range(new_shape[0]):
      for col in range(new_shape[1]):
        new_mask[row, col, 0] = old_mask[row, col]

  niftiImage = nib.Nifti1Image(new_mask, affine=affinity)
  nib.save(niftiImage, os.path.join(dataset, "imagesTs", mask_name))

print(len(masks_wrong_size) == count)


