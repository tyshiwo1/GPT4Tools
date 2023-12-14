import json
import os 

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

with open(os.path.join('./', caption_path), 'w') as f:
    annotations = json.dump(annotations, f)
    f.close()