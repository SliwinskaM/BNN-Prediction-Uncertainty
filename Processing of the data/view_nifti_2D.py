import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import os
import glob

dataset = "SKB"
# specific images
# path_comps = [['imagesTs', 'ID5C0_0000.nii.gz'], ['labelsTs', 'ID5C0.nii.gz'], ['labelsTs', 'ID5C1.nii.gz'], ['labelsTs', 'ID5C2.nii.gz'], \
#               ['DROPOUT00', 'predsTs', 'ID5C0.nii.gz'], ['DROPOUT04', 'predsTs_average', 'ID5C0.nii.gz']] 
# path_comps = [['imagesTs', 'ID11C12_0000.nii.gz'], ['labelsTs', 'ID11C12.nii.gz'], ['labelsTs', 'ID11C13.nii.gz'], ['labelsTs', 'ID11C14.nii.gz'], \
#               ['DROPOUT00', 'predsTs', 'ID11C12.nii.gz'], ['DROPOUT04', 'predsTs_average', 'ID11C12.nii.gz']] 

# path_comps = [['labelsTs', 'ID5C0.nii.gz'], ['labelsTs', 'ID5C1.nii.gz'], ['labelsTs', 'ID5C2.nii.gz']]
# mean_name = "labelsTs_ID5C0-2_mean"

# path_comps = [['labelsTs', 'ID11C18.nii.gz'], ['labelsTs', 'ID11C19.nii.gz'], ['labelsTs', 'ID11C20.nii.gz']]
# mean_name = "labelsTs_ID11C18-20_mean"

path_comps = [['imagesTs', 'ID11C18_0000.nii.gz']]  # [['DROPOUT00', 'predsTs', 'ID11C18.nii.gz'], ['DROPOUT04', 'predsTs_average', 'ID11C18.nii.gz']]



# # whole test dataset
# path_comps = []
# test_labels = sorted(os.listdir(dataset + '/labelsTs'))
# for lab in test_labels:
#   path_comps.append(['labelsTs', lab])


all_images_list = []

for path_comp in path_comps:
  view_name = '_'.join(path_comp)
  path_comp = [dataset] + path_comp
  image_path = os.path.join(*path_comp)

  image = nib.load(image_path).get_fdata()

  if image.shape[2] > 1:
    raise("Not 2D!!!!!!")

  plt.imshow(- image[:,:,0], interpolation="none", cmap="Greys") 
  plt.savefig(dataset + "/view_nifti/" + view_name + '.png')

#   # calculate mean
#   all_images_list.append(image)

# all_images = np.stack(all_images_list)
# mean_image = np.mean(all_images, axis=0)
# plt.imshow(- mean_image[:,:,0], interpolation="none", cmap="Greys") 
# plt.savefig(dataset + "/view_nifti/" + mean_name + '.png')

