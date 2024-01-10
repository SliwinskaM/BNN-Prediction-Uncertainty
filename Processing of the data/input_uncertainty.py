import numpy as np
import nibabel as nib
import glob
import os
import pylidc as pl
from pylidc.utils import consensus
import torch


dataset_uncertainty_list = []

dictName = 'my_lidc_name_id_mapping.pck'
# in
imgDir = '/net/pr1/plgrid/plggonwelo/Lung/LIDC/IMAGES/' 
# # out
# casesDirCT = '/net/people/plgsliwinska/venv/project_files/CT_LESIONS_GREATER_THAN_3mm/'
# casesDirMasks = '/net/people/plgsliwinska/venv/project_files/segm_masks_LESIONS_GREATER_THAN_3mm/'
casesDirDebug = '/net/people/plgsliwinska/venv/project_files/Debug/'


namesMappingDict = torch.load(dictName)
PrometheusToLidcMapping = {}

for key,value in zip(namesMappingDict.keys(),namesMappingDict.values()):
	PrometheusToLidcMapping[value] = key

scratchNames = sorted(glob.glob(imgDir + 'IMG_*.nii.gz'))

# create positive samples

PADDING_X = 20
PADDING_Y = 20
PADDING_Z = 2
agreementLevel = 0.01
sizeTH = 3

# counter for naming the files
count = 0
count_debug = 0
globalMaxMalignacy = 0

number = 0
for scratchName in scratchNames:
  if True: # number < 4:
    number += 1

    if os.path.basename(scratchName) not in PrometheusToLidcMapping.keys():
      continue

    print("--------IMAGE: ", os.path.basename(scratchName), "---------------")
    
    LIDC_Name = PrometheusToLidcMapping[os.path.basename(scratchName)]

    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == LIDC_Name).first()

    # load image
    niftiVol = nib.load(scratchName)
    vol = niftiVol.get_fdata() # 3D matrix with an image
    affinity = niftiVol.affine
    nods = scan.cluster_annotations() # a list of cancer changes

    # outputImages = []

    for i in range(len(nods)):
      # outputImages.append([])
      anns = nods[i]

      # common mask (not used directly), common bounding box (slice for every axis), individual masks
      cmask,cbbox,masks = consensus(anns, clevel=agreementLevel,pad=[(PADDING_X,PADDING_Y), (PADDING_X,PADDING_Y), (PADDING_Z,PADDING_Z)])
        
      locations = [] # for each individual mask, every pixel in each axis that is masked
      for mask in masks: 
        locations.append(np.where(mask!=0))

      # real-life size of a lesion in the widest place
      maxSize = np.max([np.max([(np.max(loc) - np.min(loc))*s for loc,s in zip(location,[affinity[i,i] for i in range(3)])]) for location in locations])
      print("len(masks):", len(masks))

      # assure that lesions are > 3mm
      if maxSize > sizeTH:
  ##### INPUT DATA UNCERTAINTY
        masks_for_lesion_list = [] # 1 nod = 1 lesion

        # save every mask and copy of the image for every mask
        for mask in masks:
          dum = np.zeros(cmask.shape,dtype=np.uint8)
          dum += mask
          dum[dum!=0] = 1 

          masks_for_lesion_list.append(dum)

          # TU BY BYLO ZAPISYWANIE OBRAZOW DLA KAZDEJ ADNOTACJI KAZDEGO LESIONA
          ###
          niftiImageMask = nib.Nifti1Image(dum, affine=affinity)
          newNameMask = 'CT_' + ('0'*(4-len(str(count)))) + str(count) + ".nii.gz"
          nib.save(niftiImageMask, '/net/people/plgsliwinska/venv/project_files/segm_masks_LESIONS_GREATER_THAN_3mm/' + newNameMask)
          ###


          # newNameMask = 'CT_' + ('0'*(4-len(str(count)))) + str(count) + ".nii.gz"
          count += 1
          print("Image and mask:", newNameMask)

        masks_for_lesion = np.array(masks_for_lesion_list)
        print("masks_for_lesion.shape:", masks_for_lesion.shape)
        masks_zero = (masks_for_lesion == 0).astype(int)
        # print("masks_zero.shape:", masks_zero.shape)
        uncertainty_masks_zero = np.mean(masks_zero, axis=0)
        # print("uncertainty_masks_zero.shape:", uncertainty_masks_zero.shape)

        uncertainty_masks = np.zeros(uncertainty_masks_zero.shape)
        for row in range(len(uncertainty_masks)):
           for col in range(len(uncertainty_masks[0])):
              for i in range(len(uncertainty_masks[0, 0])):
                 if uncertainty_masks_zero[row, col, i] > 0.5:
                    uncertainty_masks[row, col, i] = 1 - uncertainty_masks_zero[row, col, i]
                 else:
                    uncertainty_masks[row, col, i] = uncertainty_masks_zero[row, col, i]
        
        print("uncertainty_masks.shape:", uncertainty_masks.shape)
        # print("uncertainty_masks:", uncertainty_masks)
        print(np.all(np.all(np.all(uncertainty_masks >= 0.5))))
        niftiDebug = nib.Nifti1Image(uncertainty_masks, affine=affinity)
        newNameDebug = "Debug" + str(count_debug) + ".nii.gz"
        count_debug += 1
        print("Debug name:", newNameDebug)
        nib.save(niftiDebug, casesDirDebug + newNameDebug)

        assert np.all(np.all(np.all(uncertainty_masks <= 0.5)))

        mean_uncertainty = np.mean(np.mean(np.mean(uncertainty_masks)))
        dataset_uncertainty_list.append(mean_uncertainty)


uncertainty = np.mean(dataset_uncertainty_list)

# normalize - uncertainty was on a scale from 0 to 0.5, it has to be 0-1
uncertainty_norm = uncertainty * 2
print("The uncertainty of the dataset:", uncertainty_norm)