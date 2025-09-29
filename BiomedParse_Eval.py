import json
import os
import argparse
import logging

import time
import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from my_util.my_inference import *

logger = logging.getLogger('experiment_logger')
logger.setLevel(logging.INFO)

# image_targets = BIOMED_OBJECTS['MRI-Cardiac']
image_targets = [
    'myocardium',
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

def setup_logging(cp_name, val_test, rem_overlapping_by, class_prob_thres):
    log_filename = f"/cluster/home/xueqwang/BiomedParse_ft/Dice_{cp_name}_{val_test}_{rem_overlapping_by}_{class_prob_thres}.log"

    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)

def biomedparse_inference_evaluate_slice(image, image_targets, height, width, model, lora, gt_masks, prompt_all_classes=True, rem_overlapping_by='class_prob', class_prob_thres=-1):

    gt_masks_compact = np.stack([gt_masks.get(t, np.zeros((1024, 1024)).astype(np.uint8)) for t in image_targets])

    prompt_targets = image_targets if prompt_all_classes else list(gt_masks.keys())
    
    # all prompts
    data = {"image":image, "text":prompt_targets, "height":height, "width":width}
    _, masks_prob_bf_postprocessing, masks_bf_postprocessing, classification_prob = biomedparse_inference(data, model, lora)    
    if rem_overlapping_by == 'class_prob':
        _, masks = biomedparse_inference_postprocessing_class_prob(masks_prob_bf_postprocessing, classification_prob, prompt_targets, class_prob_thres=class_prob_thres)
    else:
        _, masks = biomedparse_inference_postprocessing_sigmoid(masks_prob_bf_postprocessing, prompt_targets)
    
    masks_compact = np.stack([masks.get(t, np.zeros((1024, 1024))).astype(np.uint8) for t in image_targets])
    I = (masks_compact*gt_masks_compact).sum(axis=(1,2))
    denominator = masks_compact.sum(axis=(1,2))+gt_masks_compact.sum(axis=(1,2))

    return I, denominator, gt_masks_compact.sum(axis=(1,2))

def biomedparse_inference_evaluate_patient(
    file_name_img, file_name_seg, raw_data_folder, 
    no_slices_per_sample, slice_freq,
    image_targets, 
    model, lora, 
    prompt_all_classes = True, 
    rem_overlapping_by='class_prob', class_prob_thres=-1):
    
    sliced_img_data, sliced_gt_data = sliced_img_gt_retrieval(file_name_img, file_name_seg, raw_data_folder, no_slices_per_sample, slice_freq)

    I_patient = np.zeros((0,len(image_targets)))
    denominator_patient = np.zeros((0,len(image_targets)))
    gt_patient = np.zeros((0,len(image_targets)))

    for s in range(sliced_img_data.shape[-1]):
        slice_index = (slice(None), slice(None), s)
        img_transformed, gt_masks, gt_mask_agg = pad_img_mask(sliced_img_data, sliced_gt_data, slice_index, gt_labels=hvsmr_labels, intensify=True)
         
        img_transformed = np.stack([img_transformed]*3, axis=-1)
    
        width, height = img_transformed.shape[0], img_transformed.shape[1]
        img_transformed = np.asarray(img_transformed)
        image = torch.from_numpy(img_transformed.copy()).permute(2,0,1).cuda()

        I, denominator, gt = biomedparse_inference_evaluate_slice(
            image, image_targets, height, width, 
            model, lora, gt_masks, 
            prompt_all_classes=prompt_all_classes, rem_overlapping_by=rem_overlapping_by, class_prob_thres=class_prob_thres)

        I_patient = np.vstack([I_patient, I.T])
        denominator_patient = np.vstack([denominator_patient, denominator.T])
        gt_patient = np.vstack([gt_patient, gt.T])

    return I_patient, denominator_patient, gt_patient

def biomedparse_inference_evaluate_model(
    file_names_img_gt, raw_data_folder,
    no_slices_per_sample, slice_freq,
    image_targets, 
    model, lora,
    prompt_all_classes = True, rem_overlapping_by='class_prob',
    class_prob_thres = -1
):

    I_model = []
    denominator_model = []
    gt_model = []

    for file_name_img, file_name_seg in file_names_img_gt:
        I, D, gt = biomedparse_inference_evaluate_patient(
            file_name_img, file_name_seg, raw_data_folder,
            no_slices_per_sample, slice_freq,
            image_targets,
            model, lora,
            prompt_all_classes=prompt_all_classes, 
            rem_overlapping_by=rem_overlapping_by, class_prob_thres=class_prob_thres
        )

        I_model.append(I)
        denominator_model.append(D)
        gt_model.append(gt)

    return I_model, denominator_model, gt_model

def dice_summary(I, D, gt, image_targets):
    I_slices, D_slices, gt_slices = np.vstack(I), np.vstack(D), np.vstack(gt)
    dice_slices = 2*I_slices/D_slices
    gt_freq_slices = gt_slices/(1024**2)

    I_patient = np.vstack([i.sum(axis=0).T for i in I])
    D_patient = np.vstack([d.sum(axis=0).T for d in D])
    dice_patient = 2*I_patient/D_patient
    gt_freq_patient = [gt_p.sum(axis=0)/(gt_p.shape[0]*(1024**2)) for gt_p in gt]

    I_model, D_model = I_patient.sum(axis=0), D_patient.sum(axis=0)
    dice_model = I_model*2/D_model
    gt_freq_model = gt_slices.sum(axis=0)/(gt_slices.shape[0]*(1024**2))

    logger.info(f"  Image Targets: {image_targets}")
    logger.info(f"  Area by class (per slice)     : {np.round(np.nanmean(gt_freq_slices, axis=0)*100, 1)}")
    logger.info(f"  Area by class (per patient)   : {np.round(np.nanmean(gt_freq_patient, axis=0)*100, 1)}")
    logger.info(f"  Area by class (all test cases): {np.round(100*gt_freq_model, 1)}")
    logger.info("")
    logger.info(f"  Dice by class (per slice)     : {np.round(np.nanmean(dice_slices, axis=0)*100, 1)}. mDice overall: {np.round(np.nanmean(dice_slices)*100, 1)}. cDice overall: {np.round(np.nanmean(2*I_slices.sum(axis=1)/D_slices.sum(axis=1))*100, 1)}")
    logger.info(f"  Dice by class (per patient)   : {np.round(np.nanmean(dice_patient, axis=0)*100, 1)}. mDice overall: {np.round(np.nanmean(dice_patient)*100, 1)}. cDice overall: {np.round(np.nanmean(2*I_patient.sum(axis=1)/D_patient.sum(axis=1))*100, 1)}")
    logger.info(f"  Dice by class (all test cases): {np.round(100*dice_model, 1)}. mDice overall: {np.round(np.nanmean(dice_model)*100, 1)}. cDice overall: {np.round(np.nanmean(2*I_model.sum()/D_model.sum())*100, 1)}")

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cp_name', type=str, required=True, 
                       help='name of the checkpoint')
    parser.add_argument('--val_test', type=str, required=True, 
                       help='Evaluation dataset. Choose between val or test')
    parser.add_argument('--rem_overlapping_by', type=str, required=True, 
                       help='Method for removing overlapping masks. Choose between class_prob or avg_pixel_sigmoids')
    parser.add_argument('--class_prob_thres', type=float, required=True, 
                       help='threshold of class probability. -1 for no threshold, (0,1) for threshold')

    args = parser.parse_args()

    setup_logging(args.cp_name, args.val_test, args.rem_overlapping_by, args.class_prob_thres)

    exp_name = args.cp_name
    lora = 'lora' in exp_name

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
    val_samples = [8, 11, 24, 29, 59]
    # val_samples = [58, 56, 49, 17, 1]
    test_samples = [0, 10, 12, 16, 19, 30, 40, 45, 50, 53]
    # test_samples = [12,45,10,40,50,53,30,16,0,19,24,11,8,29,59,51,21,39,22,47,18,31,28,42,46]
    no_slices_per_sample = 0
    slice_freq = 1

    if args.val_test == 'test':
        file_names = [(f'pat{str(p)}_cropped.nii.gz', f'pat{str(p)}_cropped_seg.nii.gz') for p in test_samples]
    else:
        file_names = [(f'pat{str(p)}_cropped.nii.gz', f'pat{str(p)}_cropped_seg.nii.gz') for p in val_samples]
    
    I, D, gt = biomedparse_inference_evaluate_model(file_names, raw_data_folder, 
                                                    no_slices_per_sample, slice_freq, 
                                                    image_targets[1:], model, lora=lora, 
                                                    prompt_all_classes=True, 
                                                    rem_overlapping_by=args.rem_overlapping_by, class_prob_thres=args.class_prob_thres)
    logger.info(f"Arguments:")
    logger.info(f"  Checkpoint: {exp_name}")
    logger.info(f"  Evaluate on: {args.val_test}")
    logger.info(f"  Remove Overlapping Masks by: {args.rem_overlapping_by}")
    logger.info(f"  Class Probability Threshold: {args.class_prob_thres}")
    dice_summary(I, D, gt, image_targets[1:])

    end = time.time()

    logger.info(f"Execution Time: {datetime.timedelta(seconds=end - start)}")

if __name__ == "__main__":
    main()

