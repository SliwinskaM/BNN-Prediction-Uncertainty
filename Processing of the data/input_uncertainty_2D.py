import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import re

dataset = "SKB"

train_labels_unsorted = ['labelsTr/' + img_name for img_name in sorted(os.listdir(dataset + '/labelsTr'))]
test_labels_unsorted = ['labelsTs/' + img_name for img_name in sorted(os.listdir(dataset + '/labelsTs'))]

# sort according to the file names treated as numbers
ordered_labels = []
for first_number in range(40):
   for second_number in range(300):
      pattern1 = "labelsTr/ID" + str(first_number) + 'C' + str(second_number) + ".nii.gz"
      pattern2 = "labelsTs/ID" + str(first_number) + 'C' + str(second_number) + ".nii.gz"

      for tr_label in train_labels_unsorted:
        if tr_label == pattern1:
           ordered_labels.append(tr_label)

      for ts_label in test_labels_unsorted:
        if ts_label == pattern2:
           ordered_labels.append(ts_label)



# check if all of the labels are included
print(len(ordered_labels))
print(len(train_labels_unsorted))
print(len(test_labels_unsorted))
# assert len(ordered_labels) == len(train_labels_unsorted) + len(test_labels_unsorted)



dataset_uncertainty_list = []


no_of_groups = int(len(ordered_labels) / 3)
print("Number of groups:", no_of_groups)

for group_no in range(no_of_groups):
    print(group_no, "---------------")
    group_names = [ordered_labels[group_no * 3 + i] for i in range(3)]
    print(group_names)

    masks_for_lesion_list = []
    masks_for_lesion_sizes = []

    for mask_name in group_names:
      # load label
      mask_path = os.path.join(dataset, mask_name)
      mask_nifti = nib.load(mask_path)
      mask = mask_nifti.get_fdata() # 3D numpy matrix with an image
      affinity = mask_nifti.affine
      # print("mask.shape:", mask.shape)

      masks_for_lesion_list.append(mask)
      masks_for_lesion_sizes.append(mask.size)

      # check if masks are for the same image
    # print(masks_for_lesion_sizes)
    assert (masks_for_lesion_sizes[0] == masks_for_lesion_sizes[1] == masks_for_lesion_sizes[2])


    ##### INPUT DATA UNCERTAINTY
    masks_for_lesion = np.array(masks_for_lesion_list)
    # print("masks_for_lesion.shape:", masks_for_lesion.shape)
    masks_zero = (masks_for_lesion == 0).astype(int)
    # print("masks_zero.shape:", masks_zero.shape)
    uncertainty_masks_zero = np.mean(masks_zero, axis=0)

    # input images can have different dimensions - niektóre są 3D z ostatnim wymiarem o długości 1
    # print("uncertainty_masks_zero.shape:", uncertainty_masks_zero.shape)
    if len(uncertainty_masks_zero.shape) == 3:
       uncertainty_masks_zero = uncertainty_masks_zero[:,:,0]
    # print("uncertainty_masks_zero.shape:", uncertainty_masks_zero.shape)

    uncertainty_masks = np.zeros((uncertainty_masks_zero.shape[0], uncertainty_masks_zero.shape[1]))
    for row in range(len(uncertainty_masks)):
        for col in range(len(uncertainty_masks[0])):
              if uncertainty_masks_zero[row, col] > 0.5:
                uncertainty_masks[row, col] = 1 - uncertainty_masks_zero[row, col]
              else:
                uncertainty_masks[row, col] = uncertainty_masks_zero[row, col]
    
    assert np.all(np.all(np.all(uncertainty_masks <= 0.5)))
    # assert len(uncertainty_masks.shape) == 2
    # print("uncertainty_masks.shape:", uncertainty_masks.shape)


    ##### save debugging image and corresponding masks and original image
    if group_no in [0, 50, 100, 150, 200, 250, 300]:
      # debug
      newNameDebug = "Debug" + str(group_no)
      print("Debug name and shape:", newNameDebug, uncertainty_masks.shape)
      plt.imshow(- uncertainty_masks, interpolation="none", cmap="Greys") 
      plt.savefig(dataset + "/view_nifti/" + newNameDebug + '.png')

      # original
      org_path = os.path.join(dataset, group_names[0]).replace("labels", "images").replace(".nii", "_0000.nii")
      org = nib.load(org_path).get_fdata()
      print("Original image and shape:", org_path, org.shape)
      plt.imshow(- org, interpolation="none", cmap="Greys") 
      plt.savefig(dataset + "/view_nifti/" + newNameDebug + '_Org.png')

      # masks
      for i in range(len(group_names)):
        mask_path = os.path.join(dataset, group_names[i])
        mask = nib.load(mask_path).get_fdata()
        print("Mask name and shape:", mask_path, mask.shape)
        plt.imshow(- mask, interpolation="none", cmap="Greys") 
        plt.savefig(dataset + "/view_nifti/" + newNameDebug + '_Mask' + str(i) + '.png')


    mean_uncertainty = np.mean(np.mean(np.mean(uncertainty_masks)))
    dataset_uncertainty_list.append(mean_uncertainty)


uncertainty = np.mean(dataset_uncertainty_list)

# normalize - uncertainty was on a scale from 0 to 0.5, it has to be 0-1
uncertainty_norm = uncertainty * 2
print("The uncertainty of the dataset:", uncertainty_norm)