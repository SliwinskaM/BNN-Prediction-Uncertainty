#!/usr/bin/env python
# coding: utf-8

# https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours,label,regionprops
import nibabel as nib
import glob
import os
from pycimg import CImg
import pylidc as pl
from pylidc.utils import consensus
import torch
import csv


dictName = 'my_lidc_name_id_mapping.pck'
# in
imgDir = '/net/pr1/plgrid/plggonwelo/Lung/LIDC/IMAGES/' 
# out
casesDirCT = '/net/people/plgsliwinska/venv/project_files/CT_LESIONS_GREATER_THAN_3mm/'
casesDirMasks = '/net/people/plgsliwinska/venv/project_files/segm_masks_LESIONS_GREATER_THAN_3mm/'
# file to save info which input image (whole organ) translates to which output images (lesions)
organToLesions = open('organToLesionsCheck.csv', 'w')
writer = csv.writer(organToLesions)


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
globalMaxMalignacy = 0

for scratchName in scratchNames:
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

  outputImages = []

  # print("Image shape:", vol.shape)
  # print("affinity:", affinity)
  # print("len(nods):", len(nods))
  # print("nods:", nods)

  for i in range(len(nods)):
    outputImages.append([])
    anns = nods[i]

    # common mask (not used directly), common bounding box (slice for every axis), individual masks
    cmask,cbbox,masks = consensus(anns, clevel=agreementLevel,pad=[(PADDING_X,PADDING_Y), (PADDING_X,PADDING_Y), (PADDING_Z,PADDING_Z)])
      
    locations = [] # for each individual mask, every pixel in each axis that is masked
    for mask in masks: 
      locations.append(np.where(mask!=0))
        

    # real-life size of a lesion in the widest place
    maxSize = np.max([np.max([(np.max(loc) - np.min(loc))*s for loc,s in zip(location,[affinity[i,i] for i in range(3)])]) for location in locations])
    print("len(anns): ", len(anns))
    print("anns: ", anns)
    print("len(masks):", len(masks))
    # print("len(masks[0]):", len(masks[0]))
    # # print("cmask:", cmask)
    # print("cmask.shape:", cmask.shape)
    # print("cbbox:", cbbox)
    # print("vol[cbbox].shape:", vol[cbbox].shape)
    # print("len(locations):", len(locations))
    # # print("locations:", locations)
    # print("maxSize:", maxSize)

    # assure that lesions are > 3mm
    if maxSize > sizeTH:
      # save the part of the CT scan representing the lesion
      niftiImageCT = nib.Nifti1Image(vol[cbbox], affine=affinity)

      # assign mmaximal malignancy
      maxMalignancy = np.max([ann.malignancy for ann in anns])
      globalMaxMalignacy = np.max([globalMaxMalignacy, maxMalignancy])
      # print("maxMalignancy:", maxMalignancy)

      # save every mask and copy of the image for every mask
      for mask in masks:
        dum = np.zeros(cmask.shape,dtype=np.uint8)
        dum += mask
        dum[dum!=0] = 1 # maxMalignancy

        niftiImageMask = nib.Nifti1Image(dum, affine=affinity)
        newNameMask = 'CT_' + ('0'*(4-len(str(count)))) + str(count) + ".nii.gz"
        nib.save(niftiImageMask, casesDirMasks + newNameMask)

        newNameCT = 'CT_' + ('0'*(4-len(str(count)))) + str(count) + "_0000.nii.gz"
        nib.save(niftiImageCT, casesDirCT + newNameCT)  

        count += 1
        outputImages[i].append(newNameCT)

        print("Saving mask:", newNameMask)
        # print("Saving image:", newNameCT)

  writer.writerow([os.path.basename(scratchName), outputImages])
  print("Saving row:", os.path.basename(scratchName), outputImages)

organToLesions.close()
print("globalMaxMalignacy:", globalMaxMalignacy)



