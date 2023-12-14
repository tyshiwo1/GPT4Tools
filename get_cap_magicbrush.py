import json
import os
from image_caption import get_blip, get_blip_res
from PIL import Image
from tqdm import tqdm
import torch
import argparse
parser = argparse.ArgumentParser(description="Get Caption")
parser.add_argument("--dev", type=str, default="dev")  
parser.add_argument("--global-descriptions", type=str, default="global_descriptions.json")  
parser.add_argument("--local-descriptions", type=str, default="local_descriptions.json")  
parser.add_argument("--caption-path", type=str, default="caption_path.json")  
args = parser.parse_args()

dev = args.dev
global_descriptions = args.global_descriptions
local_descriptions = args.local_descriptions
caption_path = args.caption_path



if dev == 'train':

    blip2_processor, blip2 = get_blip()

    def to_image(tensor, is_uint8=True):
        tensor = tensor.squeeze(0).permute(1, 2, 0).contiguous()
        if tensor.shape[-1] == 1:
            tensor_fill = tensor.new_zeros(*tensor.shape[:2], 3)
            tensor_fill[..., 0:1] = tensor*(tensor < 0) * -1
            tensor_fill[..., 1:2] = tensor*(tensor > 0) * 1
            tensor = tensor_fill
        
        arr = tensor.detach().cpu().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr * 255
        if is_uint8:
            return arr.astype('uint8')
        else:
            return arr

    def get_cap(sample2_path): 
        sample2_pil = Image.open(sample2_path)
        with torch.no_grad():
            
            blip2_out = get_blip_res(
                sample2_pil, blip2_processor, blip2, device='cuda', dtype=sample2.dtype, prompt=prompt)
            textual_inversion_prompt = blip2_out
            prompt = blip2_out
        #     torch.cuda.empty_cache()

        # import gc
        # gc.collect()
        return prompt

    with open(os.path.join('./', dev, 'edit_sessions.json')) as f:
        cap_dict = json.load(f)
        f.close()

    annotations = []
    for k_name, instruct_list in tqdm(cap_dict.items()):
        for cap in instruct_list:
            sample2_path = cap["input"]
            img_number = sample2_path.split('-')[0]
            prompt = get_cap( os.path.join('./', dev, 'images', img_number, sample2_path) )
            annotation = dict()
            annotation['file_name'] = sample2_path
            annotation['caption'] = prompt
            annotations.append( annotation )
else:


    with open(os.path.join('./', dev, global_descriptions)) as f:
        cap_dict = json.load(f)
        f.close()

    with open(os.path.join('./', dev, local_descriptions)) as f:
        short_cap_dict = json.load(f) 
        f.close()

    annotations = []
    for k_name, img_cap_dict in cap_dict.items():
        for img_name, cap in img_cap_dict.items():
            print(img_name)
            annotation = dict()
            if img_name.split('.')[-2].split('-')[-1] == 'input' :
                
                # annotation['annotations'] = #img_name.split('-')[0]
                annotation['file_name'] = img_name
                annotation['caption'] = cap
                print(annotation)
                annotations.append( annotation )
            else:
                idx_output = img_name.split('.')[-2].split('-')[-1][-1]
                idx_output = int(idx_output)

                annotation['file_name'] = img_name
                annotation['caption'] = short_cap_dict[k_name][img_name]
                annotations.append( annotation )

ori_annotations = []
if os.path.exists(os.path.join('./', caption_path)):
    with open(os.path.join('./', caption_path), 'r') as f:
        ori_annotations = json.load(f)
        f.close()

with open(os.path.join('./', caption_path), 'w') as f:
    annotations = json.dump(ori_annotations+annotations, f)
    f.close()