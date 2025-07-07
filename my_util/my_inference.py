import json

import os
from collections import defaultdict
import nibabel as nib
import numpy as np
from scipy import stats
import re

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from inference_utils.processing_utils import process_intensity_image
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES, BIOMED_OBJECTS
from inference_utils.output_processing import mask_stats, combine_masks, get_target_dist, check_mask_stats
from modeling.language.loss import vl_similarity

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

colors_list = [(np.array(color['color'])).tolist() for color in COCO_CATEGORIES]
color_codes = [f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}' for color in colors_list]

colors_selected = [0,2,4,16,12,9,10,58,96]
sg_targets = [
    'myocardium',
    'left heart ventricle',
    'right heart ventricle',
    'left heart atrium',
    'right heart atrium',
    'aorta',
    'pulmonary artery',
    'superior vena cava',
    'inferior vena cava'
]
# image_targets = BIOMED_OBJECTS['MRI-Cardiac']
targets_color_dict = {t:c for t, c in zip(sg_targets, colors_selected)}

def process_mask(gt_seg, label_target_dict, targets_color_dict=targets_color_dict, colors_list=colors_list):
    """
    Suitable only for mask with classes encoded as integers 1, 2, 3, ... (not binary masks stored in a dictionary)
    
    Args:
        - gt_seg: ground truth segmentations tensor
        - label_target_dict: {class_num: class_name}
        - targets_color_dict = targets_color_dict
        - colors_list = colors_list

    Return:
        mask: 3-D tensor. Length of last axis = 4: first 3 corresponds to the RGB code of the color of corresponding class, last digit = transparency (128 out of 256)  
    """
    mask = np.zeros((*gt_seg.shape, 4), dtype=int)
    seg_classes = list(np.unique(gt_seg))
    for c in seg_classes:
        if c>0:
            mask_color = targets_color_dict[label_target_dict[c]]
            mask[gt_seg==c] = colors_list[mask_color]+[128]
    return mask

def non_maxima_suppression(masks, p_values):
    """
    Filter out overlapping masks with lower p-values.
    masks: a dictionary of masks, {TARGET: mask}
    p_values: a dictionary of p-values, {TARGET: p_value}
    """
    mask_list = [(target, mask, p_values[target]) for target, mask in masks.items()]
    mask_list.sort(key=lambda x: x[2], reverse=True)
    
    nms_masks = {}
    for target, mask, p_value in mask_list:
        mask_region = 1*(mask > 0.5)
        if mask_region.sum() == 0:
            continue
        mask_area = mask_region.sum()
        mask_overlap = False
        for t in nms_masks:
            overlap_area = (mask_region * (nms_masks[t]>0.5)).sum()
            if overlap_area > 0.5 * mask_area:
                mask_overlap = True
                break
        if not mask_overlap:
            nms_masks[target] = mask
    return nms_masks

def non_maxima_suppression_sigmoids(masks, sigmoids):
    """
    Alternative of original non_maxima_suppression in BiomedParse. Using sigmoids instead of p-values, given that p-values are not available for extended ontologies

    Filter out overlapping masks with lower average sigmoids.
    masks: a dictionary of masks, {TARGET: mask}
    sigmoids: a dictionary of sigmoids, {TARGET: [sigmoids_sum, sigmoids_mean]}
    """
    mask_list = [(target, mask, sigmoids[target][1]) for target, mask in masks.items() if sigmoids[target][0]>0]
    mask_list.sort(key=lambda x: x[2], reverse=True)
    
    nms_masks = {}
    for target, mask, sigmoids in mask_list:
        mask_region = 1*(mask > 0.5)
        if mask_region.sum() == 0:
            continue
        mask_area = mask_region.sum()
        mask_overlap = False
        for t in nms_masks:
            overlap_area = (mask_region * (nms_masks[t]>0.5)).sum()
            if overlap_area > 0.5 * mask_area:
                mask_overlap = True
                break
        if not mask_overlap:
            nms_masks[target] = mask
    return nms_masks

def biomedparse_inference_data_prep(img_data, slice_idx, transform_func):
    """
    Pad and upsample image_data to ensure size = 1024 x 1024

    Return:
        - width, height
        - image_resize: 1024 x 1024 image
        - img_data_slice: permuted image_resize as torch tensor and sent to cuda, ready to be used for inference
    """
    img_data_slice = process_intensity_image(img_data[slice_idx], is_CT=False)
    image_resize = transform_func(Image.fromarray(img_data_slice))
    
    width = image_resize.size[0]
    height = image_resize.size[1]
    image_resize = np.asarray(image_resize)
    img_data_slice = torch.from_numpy(image_resize.copy()).permute(2,0,1).cuda()

    return width, height, image_resize, img_data_slice


def biomedparse_inference_masks(data, results, extra, model):
    """
    Inference

    Return:
        - pred_mask_prob [torch tensor]: predicted logits
        - pred_mask_pos [torch tensor]: binary mask derived from pred_mask_prob with threshold = 0.5
    """
    pred_masks = results['pred_masks'][0]
    v_emb = results['pred_captions'][0]
    t_emb = extra['grounding_class']
    
    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
    
    temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

    match_id = out_prob.max(0)[1]
    pred_masks_pos = pred_masks[match_id, :, :]

    pred_mask_prob = F.interpolate(pred_masks_pos[None,], (data['height'], data['width']), 
                                   mode='bilinear')[0,:,:data['height'],:data['width']].sigmoid().cpu().numpy()
    pred_masks_pos = (1*(pred_mask_prob > 0.5)).astype(np.uint8)

    return pred_mask_prob, pred_masks_pos

def biomedparse_inference_all_prompts(width, height, image_resize, img_data_slice, image_targets, model, remove_overlapping_by = 'p-val'):
    """
    Recognition by performing inference once with all prompts

    Args:
        - width, height, image_resize, image_data_slice: outputs from biomedparse_inference_data_prep
        - image_targets: list of class names (by default: all ontologies of heart)
        - model: model to be used for inference
        - remove_overlapping_by: statistics used to remove overlapping masks. Default: p-val. Alternative: sigmoids

    Return:
        - predicts_raw: all predictions (logits) before removing overlapping masks
        - masks: final binary mask saved in dictionary
    """
    data = {"image":img_data_slice, "text":image_targets, "height":height, "width":width}
    batched_inputs = [data]
    
    with torch.no_grad():
        results, image_size, extra = model.model.evaluate_demo(batched_inputs)
    pred_mask_prob, pred_masks_pos = biomedparse_inference_masks(data, results, extra, model)

    predicts = {}
    p_values = {}
    sigmoid_avg = {}
    
    for i, t in enumerate(image_targets):
        predicts[t] = pred_mask_prob[i]
        if t in BIOMED_OBJECTS['MRI-Cardiac']:
            adj_p_value = check_mask_stats(image_resize, pred_mask_prob[i]*255, 'MRI-Cardiac', t)
        else:
            adj_p_value = 0
        p_values[t] = adj_p_value
        sigmoid_avg[t] = [pred_mask_prob[i][pred_mask_prob[i]>0.5].sum(), pred_mask_prob[i][pred_mask_prob[i]>0.5].mean()]
    predicts_raw = predicts.copy()
    if remove_overlapping_by == 'p-val':
        predicts = non_maxima_suppression(predicts, p_values)
    else:
        predicts = non_maxima_suppression_sigmoids(predicts, sigmoid_avg)
    masks = combine_masks(predicts)

    return predicts_raw, masks


def biomedparse_inference_single_prompt(width, height, image_resize, img_data_slice, image_targets, model, remove_overlapping_by = 'p-val'):
    """
    Recognition by performing inference iteratively with one prompt a time

    Args:
        - width, height, image_resize, image_data_slice: outputs from biomedparse_inference_data_prep
        - image_targets: list of class names (by default: all ontologies of heart)
        - model: model to be used for inference
        - remove_overlapping_by: statistics used to remove overlapping masks. Default: p-val. Alternative: sigmoids

    Return:
        - predicts_raw: all predictions (logits) before removing overlapping masks
        - masks: final binary mask saved in dictionary
    """
    predicts = {}
    p_values = {}
    sigmoid_avg = {}
    for i, batch_targets in enumerate(image_targets):
        data = {"image":img_data_slice, "text":[batch_targets], "height":height, "width":width}
        batched_inputs = [data]

        with torch.no_grad():
            results, image_size, extra = model.model.evaluate_demo(batched_inputs)
        pred_mask_prob, pred_masks_pos = biomedparse_inference_masks(data, results, extra, model)

        if batch_targets in BIOMED_OBJECTS['MRI-Cardiac']:
            adj_p_value = check_mask_stats(image_resize, pred_mask_prob[0]*255, 'MRI-Cardiac', batch_targets)
        else:
            adj_p_value = 0
        # adj_p_value = 1

        predicts[batch_targets] = pred_mask_prob[0]
        p_values[batch_targets] = adj_p_value
        sigmoid_avg[batch_targets] = [pred_mask_prob[0][pred_mask_prob[0]>0.5].sum(), pred_mask_prob[0][pred_mask_prob[0]>0.5].mean()]
    predicts_raw = predicts.copy()
    if remove_overlapping_by == 'p-val':
        predicts = non_maxima_suppression(predicts, p_values)
    else:
        predicts = non_maxima_suppression_sigmoids(predicts, sigmoid_avg)
    masks = combine_masks(predicts)

    return predicts_raw, masks


def process_biomedparse_mask(pred_seg, targets_color_dict=targets_color_dict, colors_list=colors_list):
    """
    Suitable only for binary masks stored in a dictionary
    
    Args:
        - pred_seg: binary masks
        - targets_color_dict = targets_color_dict
        - colors_list = colors_list

    Return:
        mask: 3-D tensor. Length of last axis = 4: first 3 corresponds to the RGB code of the color of corresponding class, last digit = transparency (128 out of 256)  
    """
    mask = np.zeros((*list(pred_seg.values())[0].shape, 4), dtype=int)
    for i, s in enumerate(list(pred_seg.keys())):
        mask_color = targets_color_dict[s]
        mask[pred_seg[s]>0.5] = colors_list[mask_color]+[128]
    return mask

def load_model(pretrained_pth):
    """
    load model (IN EVALUATION MODE) using corresponding checkpoint
    """
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    if os.environ.get('JPY_SESSION_NAME','') == '':
        opt = init_distributed(opt)
    else:
        opt['CUDA'] = opt.get('CUDA', True) and torch.cuda.is_available()
        opt['env_info'] = 'no MPI'
        opt['world_size'] = 1
        opt['local_size'] = 1
        opt['rank'] = 0
        opt['local_rank'] = 0
        opt['master_address'] = '127.0.0.1'
        opt['master_port'] = '8673'
        torch.cuda.set_device(opt['local_rank'])
        opt['device'] = torch.device("cuda", opt['local_rank'])

    # opt = load_opt_from_config_files(["configs/biomed_seg_lang_v1.yaml"])
    # opt['TEST']['BATCH_SIZE_TOTAL'] = 1
    # opt['FP16'] = True
    # opt['WEIGHT'] = True
    # opt['STANDARD_TEXT_FOR_EVAL'] = True
    # opt['RESUME_FROM'] = '/cluster/customapps/biomed/grlab/users/xueqwang/hf_models/'
    # opt = init_distributed(opt)
    
    # pretrained_pth = '/cluster/customapps/biomed/grlab/users/xueqwang/hf_models/microsoft/biomedparse_v1.pt'
    # model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).train().cuda()
    
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

    return model