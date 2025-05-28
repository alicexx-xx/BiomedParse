import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import json

import os
os.chdir('/cluster/customapps/biomed/grlab/users/xueqwang/BiomedParse')

import sys
sys.path.append('/cluster/customapps/biomed/grlab/users/xueqwang/BiomedParse')

from collections import defaultdict
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from scipy import stats
import re

# %matplotlib ipympl

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
from inference_utils.inference import interactive_infer_image_all, non_maxima_suppression
from inference_utils.output_processing import mask_stats, combine_masks, get_target_dist, check_mask_stats

from modeling.language.loss import vl_similarity

from time import gmtime, strftime

import tqdm

t = []
t.append(transforms.Resize((1024, 1024), interpolation=Image.BICUBIC))
transform = transforms.Compose(t)

def biomedoarse_inference_data_prep(img_data, slice_idx, transform_func):
    img_data_slice = process_intensity_image(img_data[slice_idx], is_CT=False)
    image_resize = transform_func(Image.fromarray(img_data_slice))
    
    width = image_resize.size[0]
    height = image_resize.size[1]
    image_resize = np.asarray(image_resize)
    img_data_slice = torch.from_numpy(image_resize.copy()).permute(2,0,1).cuda()

    return width, height, image_resize, img_data_slice

def biomedparse_inference_masks(data, results, extra, model):
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

def biomedparse_inference_all_prompts(width, height, image_resize, img_data_slice, image_targets, model):
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
    predicts = non_maxima_suppression(predicts, p_values)
    masks = combine_masks(predicts)

    return masks, sigmoid_avg

def biomedparse_inference_single_prompt(width, height, image_resize, img_data_slice, image_targets, model):
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
    predicts = non_maxima_suppression(predicts, p_values)
    masks = combine_masks(predicts)

    return masks, sigmoid_avg

if __name__ == "__main__":
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)

    pretrained_pth = '/cluster/customapps/biomed/grlab/users/xueqwang/hf_models/microsoft/biomedparse_v1.pt'
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

    biobank_sax_labels = ['na','LV','MYO','RV']
    data_folder = '/cluster/work/grlab/projects/projects2024-ukb_cvd/D4CVD'
    sax_folder = '{0}/{1}'.format(data_folder, 'sax/data')
    image_targets = BIOMED_OBJECTS['MRI-Cardiac']


    patients = os.listdir(sax_folder)
    no_sax = len(patients)

    print(f"Inference Start: {strftime('%Y-%m-%d %H:%M:%S', gmtime())}")

    result = {}

    for s in tqdm(range(0, no_sax, 100)):
        patient = patients[s]
        sax_sample = '{0}/{1}/{2}'.format(sax_folder,patient,'sa.nii.gz')
        sax_sample_anno = '{0}/{1}/{2}'.format(sax_folder,patient,'seg_sa.nii.gz')

        result.update({patient:{}})

        img = nib.load(sax_sample)
        data = img.get_fdata()

        img_anno = nib.load(sax_sample_anno)
        data_anno_ukbb_cardiac = img_anno.get_fdata()

        # middle slice at the last timepoint
        img_shape = img.shape
        slice_idx = (slice(None), slice(None), img_shape[2]//2, img_shape[-1]-1)

        width, height, image_resize, img_data_slice = biomedoarse_inference_data_prep(data, slice_idx, transform)
        masks_all_prompts_together, sigmoids_all_prompts = biomedparse_inference_all_prompts(width, height, image_resize, img_data_slice, image_targets, model)
        masks_single_prompt, sigmoids_single_prompt = biomedparse_inference_single_prompt(width, height, image_resize, img_data_slice, image_targets, model)

        for t in image_targets:
            mask_mismatch = (masks_single_prompt[t] != masks_all_prompts_together[t]).astype(int)
            result[patient].update({
                f'{t}_Area_Single_Prompt': masks_single_prompt[t].sum(),
                f'{t}_Area_Multiple_Prompt': masks_all_prompts_together[t].sum(),
                f'{t}_Area_Mismatch': mask_mismatch.sum()
                })

    print(f"Inference Finished: {strftime('%Y-%m-%d %H:%M:%S', gmtime())}. No. of Inferences: {(no_sax//100+1)*2}")
    with open('/cluster/work/grlab/projects/tmp_xueqwang/inference_mismatch_biomedparse/patients_mismatch.json', 'w') as f:
        json.dump(result, f)

