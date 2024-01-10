#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
import torch
import nibabel as nib
from pycimg import CImg
from copy import deepcopy
from typing import Tuple, Union, List
import numpy as np
import shutil
import importlib
import pkgutil
import json
import pickle
import glob

from batchgenerators.utilities.file_and_folder_operations import join, isdir
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.utilities.file_and_folder_operations import *

from inference.predictor import Predictor


def predict_cases(plan_file,architecture_name,checkpoints, stage, output_folder, input_filenames, output_filenames, 
                  do_tta=True, step_size=0.5,overwrite=False):

    
    LABEL_LEFT = 2
    LABEL_RIGHT = 3
    LABELS = [LABEL_LEFT,LABEL_RIGHT]

    assert len(input_filenames) == len(output_filenames)

    print('#################################################')
    print("emptying cuda cache")
    torch.cuda.empty_cache()

    trainer = Predictor(plan_file, architecture_name,stage, False, True)
    trainer.process_plans(load_pickle(plan_file))

    trainer.output_folder = output_folder
    trainer.output_folder_base = output_folder
    trainer.initialize(False)

    params = [torch.load(i, map_location=torch.device('cpu')) for i in checkpoints]

    print('Trainer params')
    print(trainer.normalization_schemes, trainer.use_mask_for_norm,trainer.transpose_forward, trainer.intensity_properties)
    print('#####################################')

    if 'segmentation_export_params' in trainer.plans.keys():
        force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
        interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
        interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0


    print("starting prediction...")
    for input_filename,output_filename in zip(input_filenames, output_filenames):

        if os.path.isfile(output_filename) and overwrite==False:
            continue

        niftiSegIm =  nib.load(input_filename)
        affinity = niftiSegIm.affine
        segIm = niftiSegIm.get_fdata()
        segIm[segIm<=1] = 0

        for label in LABELS:
            dum = np.zeros(segIm.shape,dtype=type(segIm[0,0,0]))
            np.copyto(dum,segIm)
            dum[dum!=label] = 0
            dum[dum!=0] = 1
            regVoxelList = np.where(dum!=0)
            rmin = np.min(regVoxelList[0])
            rmax = np.max(regVoxelList[0])
            cmin = np.min(regVoxelList[1])
            cmax = np.max(regVoxelList[1])
            smin = np.min(regVoxelList[2])
            smax = np.max(regVoxelList[2])
            dum = dum[rmin:rmax,cmin:cmax,smin:smax]

            baseName = os.path.dirname(output_filename) + '/dumImage_0000.nii.gz'
            niftiImage = nib.Nifti1Image(dum, affine=affinity)
            nib.save(niftiImage,baseName)

            dumList = [baseName]
            d, _, dct = trainer.preprocess_patient(dumList)

            print("processing ", output_filename,d.shape)
            print("predicting", output_filename)

            #ids,counts = np.unique(d,return_counts=True)
            #print(ids,counts)
            softmax = []
            for p in params:
                trainer.load_checkpoint_ram(p, False)
                softmax.append(trainer.predict_preprocessed_data_return_seg_and_softmax(d, do_tta, trainer.data_aug_params[
                    'mirror_axes'], True, step_size=step_size, use_gaussian=True, all_in_gpu=False,mixed_precision=True)[1][None])

            softmax = np.vstack(softmax)
            softmax_mean = np.mean(softmax, 0)

            transpose_forward = trainer.plans.get('transpose_forward')
            if transpose_forward is not None:
                transpose_backward = trainer.plans.get('transpose_backward')
                softmax_mean = softmax_mean.transpose([0] + [i + 1 for i in transpose_backward])

            if hasattr(trainer, 'regions_class_order'):
                region_class_order = trainer.regions_class_order
            else:
                region_class_order = None

            name = os.path.dirname(output_filename) + '/' + str(label) + '.nii.gz'
            trainer.save_segmentation(softmax_mean, name, dct, interpolation_order, region_class_order,None, None,None, None, force_separate_z, interpolation_order_z)

            dum = nib.load(name).get_fdata()*label
            segIm[rmin:rmax,cmin:cmax,smin:smax] += dum
            segIm[segIm>=2*label] = label

            os.remove(name)
            os.remove(baseName)

        segIm[segIm>LABEL_RIGHT] = LABEL_RIGHT
        segIm[segIm<LABEL_LEFT] = 0
        niftiImage = nib.Nifti1Image(segIm, affine=affinity)
        nib.save(niftiImage,output_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",required = True, help ="json file with inference configuration")

    args = parser.parse_args()
    config_file = args.config

    with open(config_file, 'r') as config_file:
        configs = json.load(config_file)

    input_folder = configs['input_folder']
    output_folder = configs['output_folder']
    disable_tta = configs['disable_tta']
    plan_file = configs['plans_file_path']
    checkpoints = list(configs['checkpoints'].values())
    overwrite = configs['overwrite']

    maybe_mkdir_p(output_folder)


    input_files = sorted(glob.glob(input_folder + '/*.nii.gz'))
    output_files = [output_folder + '/REC_' + os.path.basename(f) for f in input_files]

    with open(configs['plans_file_path'], 'rb') as plans_file:
        plans = pickle.load(plans_file)

    network_type = configs['network_type']
    assert network_type in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], "Incorrect network type!"

    possible_stages = list(plans['plans_per_stage'].keys())

    if (network_type == '3d_cascade_fullres' or network_type == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. Run 3d_fullres.")

    if network_type == '2d' or network_type == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    if 'architecture_name' in configs.keys():
        architecture_name = configs['architecture_name']
    else:
        architecture_name = "Generic_UNet"

    # start prediction
    step_size = 0.5
    predict_cases(plan_file,architecture_name,checkpoints, stage, output_folder, input_files, output_files, not disable_tta, step_size=step_size,overwrite=overwrite)


if __name__ == "__main__":
    main()
