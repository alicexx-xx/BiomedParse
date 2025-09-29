import json
import os
import argparse
import logging

import time
import datetime
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from my_util.my_inference import *

logger = logging.getLogger('experiment_logger')
logger.setLevel(logging.INFO)

# image_targets = BIOMED_OBJECTS['MRI-Cardiac']
image_targets = [
    # 'myocardium',
    'left heart ventricle',
    'right heart ventricle',
    'left heart atrium',
    'right heart atrium',
    'aorta',
    'pulmonary artery',
    'superior vena cava',
    'inferior vena cava'
] # all ontologies under heart

hvsmr_labels = ['na','left heart ventricle','right heart ventricle','left heart atrium','right heart atrium','aorta','pulmonary artery','superior vena cava','inferior vena cava']

t = []
t.append(transforms.Resize((1024, 1024), interpolation=Image.BICUBIC))
transform_func = transforms.Compose(t)      

def setup_logging(cp_name):
    log_filename = f"/cluster/home/xueqwang/BiomedParse_ft/inference.log"

    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cp_name', type=str, required=True, 
                       help='name of the checkpoint')
    parser.add_argument('--train_val_test', type=str, required=True, 
                       help='Inference dataset. Choose between train or val or test')
    parser.add_argument('--train_size', type=int, required=True)
    parser.add_argument('--val_size', type=int, required=True)
    parser.add_argument('--test_size', type=int, required=True)

    args = parser.parse_args()

    setup_logging(args.cp_name)

    exp_name = args.cp_name
    lora = True

    inference_split = args.train_val_test
    no_sample_ft, no_sample_val, no_sample_test= args.train_size, args.val_size, args.test_size

    inference_output_folder = os.path.join('/cluster/work/grlab/projects/tmp_xueqwang/biomedparse_ft_checkpoints', exp_name, 'inference_ts')
    if not os.path.exists(inference_output_folder):
        os.makedirs(inference_output_folder, exist_ok=True)

    logger.info(f"Checkpoint: {exp_name}")
    logger.info(f"Inference split: {inference_split}")
    logger.info(f"Train Val Test (size): {(no_sample_ft, no_sample_val, no_sample_test)}")

    start = time.time()

    if lora:
        pretrained_pth = '/cluster/customapps/biomed/grlab/users/xueqwang/hf_models/microsoft/biomedparse_v1.pt'
        ft_lora_pth = f'/cluster/work/grlab/projects/tmp_xueqwang/biomedparse_ft_checkpoints/{exp_name}'
        ft_base_model_param = f'/cluster/work/grlab/projects/tmp_xueqwang/biomedparse_ft_checkpoints/{exp_name}/fine_tuned_base_model_params.safetensors'
        model = load_model(pretrained_pth, lora = True, ft_lora_pth = ft_lora_pth, ft_base_model_param=ft_base_model_param)
    else:
        pretrained_pth = f'/cluster/work/grlab/projects/tmp_xueqwang/biomedparse_ft_checkpoints/{exp_name}/model_state_dict.pt'
        model = load_model(pretrained_pth)

    raw_data_folder = '/cluster/work/grlab/projects/projects2024-ukb_cvd/D4CVD/HVSMR-2.0/cropped'

    random.seed(7)
    splits = list(range(60))
    random.shuffle(splits)

    test_samples = splits[:no_sample_test]
    val_samples = splits[no_sample_test:(no_sample_test+no_sample_val)]
    training_samples = splits[(no_sample_test+no_sample_val):]

    logger.info(f"test_samples = {','.join([str(s) for s in test_samples])}")
    logger.info(f"val_samples = {','.join([str(s) for s in val_samples])}")

    no_slices_per_sample = 0
    slice_freq = 1

    if inference_split == 'train':
        file_names = [(f'pat{str(p)}_cropped.nii.gz', f'pat{str(p)}_cropped_seg.nii.gz') for p in training_samples]
    elif inference_split == 'test':
        file_names = [(f'pat{str(p)}_cropped.nii.gz', f'pat{str(p)}_cropped_seg.nii.gz') for p in test_samples]
    else:
        file_names = [(f'pat{str(p)}_cropped.nii.gz', f'pat{str(p)}_cropped_seg.nii.gz') for p in val_samples]

    for file_name_img, file_name_seg in file_names:
        pat_name = file_name_img.split('_')[0]
        inference_res_name = os.path.join(inference_output_folder, f'{pat_name}_inference_res.npz')

        if not os.path.exists(inference_res_name):
            logger.info(f"Running inference on {pat_name}")
            sliced_img_data, sliced_gt_data = sliced_img_gt_retrieval(file_name_img, file_name_seg, raw_data_folder, no_slices_per_sample, slice_freq)
            
            shape = sliced_img_data.shape
            shape_padded = (0,0)
            if shape[0] > shape[1]:
                pad = (shape[0]-shape[1])//2
                pad_width = ((0,0), (pad, pad))
                shape_padded = (shape[0], shape[1]+2*pad)
            elif shape[0] < shape[1]:
                pad = (shape[1]-shape[0])//2
                pad_width = ((pad, pad), (0,0))
                shape_padded = (shape[0]+2*pad, shape[1])
            else:
                pad_width = None
                shape_padded = (shape[0], shape[1])

            masks_rescaled_combined_patient = np.zeros((*sliced_img_data.shape[:2],0))

            for s in range(sliced_img_data.shape[-1]):
                slice_index = (slice(None), slice(None), s)
                img_transformed, gt_masks, _ = pad_img_mask(sliced_img_data, sliced_gt_data, slice_index, gt_labels=hvsmr_labels, intensify=True)
                
                img_transformed = np.stack([img_transformed]*3, axis=-1)
                width, height = img_transformed.shape[0], img_transformed.shape[1]
                img_transformed = np.asarray(img_transformed)
                image = torch.from_numpy(img_transformed.copy()).permute(2,0,1).cuda()

                data = {"image":image, "text":image_targets, "height":height, "width":width}

                _, masks_prob_bf_postprocessing, masks_bf_postprocessing, classification_prob = biomedparse_inference(data, model, lora)
                _, masks = biomedparse_inference_postprocessing_class_prob(masks_prob_bf_postprocessing, classification_prob, image_targets, class_prob_thres=-1)

                masks_rescaled = np.stack([transform.resize(masks.get(t, np.zeros((1024, 1024))).astype(np.uint8), shape_padded, order=1, mode='constant', preserve_range=True, anti_aliasing=True) for t in image_targets])
                masks_rescaled = (masks_rescaled>=0.5).astype(np.uint8)
                if (pad_width[1] == (0,0)) and (pad_width[0] != (0,0)):
                    masks_rescaled = masks_rescaled[:,pad_width[0][0]:-pad_width[0][1],:]
                elif (pad_width[0] == (0,0)) and (pad_width[1] != (0,0)):
                    masks_rescaled = masks_rescaled[:,:,pad_width[1][0]:-pad_width[1][1]]
                masks_rescaled_combined = (masks_rescaled*np.array(range(1,len(image_targets)+1))[:,None,None]).sum(axis=0)

                masks_rescaled_combined_patient = np.concatenate([masks_rescaled_combined_patient,masks_rescaled_combined[:,:,None]], axis=-1)

            np.savez_compressed(inference_res_name, array=masks_rescaled_combined_patient)

        else:
            logger.info(f"Inference of {pat_name} already exists")

    end = time.time()

    logger.info(f"Execution Time: {datetime.timedelta(seconds=end - start)}")

if __name__ == "__main__":
    main()

