"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
from collections import OrderedDict
from monai import transforms

from lavis.datasets.datasets.base_dataset import BaseDataset
import numpy as np
import random
import torch
import json

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, extension='.nii.gz'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.vis_root = vis_root
        self.extension = extension
        # vis_root = 'data/processed_train_images'

        # self.patient_paths = [
        #     os.path.join(vis_root, f1, f2)
        #     for f1 in os.listdir(vis_root)
        #     for f2 in os.listdir(os.path.join(vis_root, f1))
        # ]

        # patient_paths = np.load('/cluster/projects/mcintoshgroup/fvlm_files/decomposed_report/patient_paths.npy') # has the resized image paths

        # the following replace the original patient_paths implementation above
        def _find_second_level_dirs(root_dir, substring=None):
            second_level_dirs = []

            for first_level in os.listdir(root_dir):
                first_path = os.path.join(root_dir, first_level)
                if os.path.isdir(first_path):
                    for second_level in os.listdir(first_path):
                        second_path = os.path.join(first_path, second_level)
                        if os.path.isdir(second_path):
                            if substring is None or substring in second_level:
                                second_level_dirs.append(second_path)

            return second_level_dirs
        
        patient_paths = _find_second_level_dirs(
            '/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_noflip_resized_train_images',
            'train_'
        )

        new_patient_paths = []
        for patient_path in patient_paths:
            new_patient_paths.append(patient_path.replace('resized_train_images', 'processed_train_images'))
        self.patient_paths = new_patient_paths

        self.organs = [
            'lung', 'heart', 'esophagus', 'aorta'
        ]

        self.loader = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
        ])

        self.organ_ratios = {k: 1 for k in self.organs}

        desc_info = json.load(open('/cluster/projects/mcintoshgroup/fvlm_files/decomposed_report/desc_info.json'))
        conc_info = json.load(open('/cluster/projects/mcintoshgroup/fvlm_files/decomposed_report/conc_info.json'))

        all_info = {}
        for patient_path in self.patient_paths:
            patient = patient_path.split('/')[-1] # the patient name like  "train_10a" or "train_10_a"
            patient = self._handle_defected_patient_dir(patient) # handle the defected directory name case

            all_info[patient] = {}
            for organ in self.organs:
                desc = ''
                if organ in desc_info.get(patient, {}):
                    desc += desc_info[patient][organ]
                    if not desc.endswith('.'):
                        desc += '.'

                conc = ''
                if organ in conc_info.get(patient, {}):
                    conc += conc_info[patient][organ]
                    if not conc.endswith('.'):
                        conc += '.'
                
                if not len(conc):
                    conc = f'{organ} shows no significant abnormalities.'
                
                input_text = conc + desc

                input_text = input_text.replace('"', '')  
                input_text = input_text.replace('\'', '')  
                input_text = input_text.replace('(', '')  
                input_text = input_text.replace(')', '')

                all_info[patient][organ] = input_text
        self.annotation = all_info

        self.crop_size = (112, 256, 352)

    def _handle_defected_patient_dir(self, patient):
        """
            from "train_10a" to "train_10_a" if neeeded 
        """
        if '_' not in patient.replace('train_', ''): # handle the defected case
            patient = patient[:-1]+'_'+patient[-1]
        return patient

    def __getitem__(self, index):
        exit = False
        while not exit:
            try:
                patient_path = self.patient_paths[index]
                choices = [file for file in os.listdir(patient_path)]
                img_path = os.path.join(patient_path, random.choice(choices))
                
                patient_id = patient_path.split('/')[-1]
                patient_id = self._handle_defected_patient_dir(patient_id)

                mask_path = img_path.replace('images', 'masks')

                data = self.loader({'image': img_path, 'label': mask_path})

                data = self.vis_processor(data)
                image = data['image'].as_tensor()
                pul_seg = data['label'][0].as_tensor()
                assert image[0].shape == self.crop_size and pul_seg.shape == self.crop_size

                text_input = self.annotation[patient_id]
                organ_abnormal_flags = torch.zeros(len(self.organs), dtype=bool)
                for i, organ in enumerate(self.organs):
                    if organ in text_input and not text_input[organ].startswith(f'{organ} shows no significant abnormalities.'):
                        organ_abnormal_flags[i] = True
                    
                    if organ not in text_input:
                        text_input[organ] = f'{organ} shows no significant abnormalities.'

                exit = True

            except Exception as e:
                print(e, patient_path)
                index = random.randint(0, len(self.patient_paths) - 1)
                continue
        
        return {
            "image": image,
            "seg": pul_seg,
            "text_input": text_input,
            "organ_abnormal_flags": organ_abnormal_flags
        }
