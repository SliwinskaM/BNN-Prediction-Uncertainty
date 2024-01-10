import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import os
import glob


debug_name = "Debug0"

if not os.path.exists('view_nifti/' + debug_name):
  os.mkdir('view_nifti/' + debug_name)

# debug
debug = nib.load('Debug/' + debug_name + '.nii.gz').get_fdata()
print("debug.shape:", debug.shape)

if not os.path.exists('view_nifti/' + debug_name):
	os.mkdir('view_nifti/' + debug_name)

for j in range(debug.shape[2]):
	# check which layers contain data
	zero = np.all(np.all(debug[:,:,j] == 0))
	if not zero:
		print(j)
		plt.imshow(debug[:,:,j]) 
		plt.savefig('view_nifti/' + debug_name + '/' + str(j) + '.png')




for image_name in ["CT_0000", "CT_0001", "CT_0002", "CT_0003"]:
  # CT
  ct = nib.load('CT_LESIONS_GREATER_THAN_3mm/' + image_name + '_0000.nii.gz').get_fdata()
  print("ct.shape:", ct.shape)

  if not os.path.exists('view_nifti/' + image_name + "_0000"):
      os.mkdir('view_nifti/' + image_name + "_0000")

  for i in range(ct.shape[2]):   
    # check which layers contain data
      zero = np.all(np.all(ct[:,:,i] == 0))
      if not zero:
        print(i)
        plt.imshow(ct[:,:,i])
        plt.savefig('view_nifti/' + image_name + '_0000/' + str(i) + '.png')


  # MASK

  mask = nib.load('segm_masks_LESIONS_GREATER_THAN_3mm/' + image_name + '.nii.gz').get_fdata()
  print("mask.shape:", mask.shape)
  mask_name = image_name

  if not os.path.exists('view_nifti/' + mask_name):
    os.mkdir('view_nifti/' + mask_name)

  for j in range(mask.shape[2]):
    zero = np.all(np.all(debug[:,:,j] == 1))
    if not zero:
      print(j)
      plt.imshow(mask[:,:,j]) 
      plt.savefig('view_nifti/' + mask_name + '/' + str(j) + '.png')



# # original
# image_name = 'IMG_0003'
# orig = nib.load('/net/pr1/plgrid/plggonwelo/Lung/LIDC/IMAGES/' + image_name + '.nii.gz').get_fdata()
# print(orig.shape)

# if not os.path.exists('view_nifti/' + image_name):
#   os.mkdir('view_nifti/' + image_name)

# for i in range(orig.shape[2]):   
# 	print(i)
# 	plt.imshow(orig[:,:,i]) 
# 	plt.savefig('view_nifti/' + image_name + '/' + str(i) + '.png')


# mask = nib.load('Original_Masks_LESIONS_GREATER_THAN_3mm/MASK_' + image_name + '.nii.gz').get_fdata()

# if not os.path.exists('view_nifti/MASK_' + image_name):
# 	os.mkdir('view_nifti/MASK_' + image_name)

# for j in range(mask.shape[2]):
# 	# check which layers contain data
# 	zero = np.all(np.all(mask[:,:,j] == 0))
# 	if not zero:
# 		print(j)
# 		plt.imshow(mask[:,:,j]) 
# 		plt.savefig('view_nifti/MASK_' + image_name + '/' + str(j) + '.png')