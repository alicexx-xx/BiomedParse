import os
os.chdir('/cluster/customapps/biomed/grlab/users/xueqwang/BiomedParse')
import sys
sys.path.append('/cluster/customapps/biomed/grlab/users/xueqwang/BiomedParse')

import json
my_result = {
    '4648154_20209_3_0': {
        'left heart ventricle_Area_Single_Prompt': 17913, 
        'left heart ventricle_Area_Multiple_Prompt': 17905, 
        'left heart ventricle_Area_Mismatch': 20, 
        'myocardium_Area_Single_Prompt': 17891, 
        'myocardium_Area_Multiple_Prompt': 17921, 
        'myocardium_Area_Mismatch': 56, 
        'right heart ventricle_Area_Single_Prompt': 14142, 
        'right heart ventricle_Area_Multiple_Prompt': 14148, 'right heart ventricle_Area_Mismatch': 8}, 
    '3153487_20209_2_0': {
        'left heart ventricle_Area_Single_Prompt': 14564, 
        'left heart ventricle_Area_Multiple_Prompt': 14562, 
        'left heart ventricle_Area_Mismatch': 2, 
        'myocardium_Area_Single_Prompt': 9062, 
        'myocardium_Area_Multiple_Prompt': 9016, 
        'myocardium_Area_Mismatch': 50, 
        'right heart ventricle_Area_Single_Prompt': 21804, 
        'right heart ventricle_Area_Multiple_Prompt': 21799, 
        'right heart ventricle_Area_Mismatch': 11}
        }

json.dump(my_result, '/cluster/work/grlab/projects/tmp_xueqwang/inference_mismatch_biomedparse/patients_mismatch.json')