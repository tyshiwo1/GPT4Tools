import json
import os 

dev = 'dev'

with open(os.path.join('./', dev, 'global_descriptions.json')) as f:
    cap_dict = json.load(f)
    f.close()

annotations = []
for k_name, img_cap_dict in cap_dict.items():
    for img_name, cap in img_cap_dict.items():
        print(img_name)
        if img_name.split('.')[-2].split('-')[-1] == 'input' :
            annotation = dict()
            
            # annotation['annotations'] = #img_name.split('-')[0]


            annotation['file_name'] = img_name
            annotation['caption'] = cap
            print(annotation)
            annotations.append( annotation )

with open(os.path.join('./', 'caption_path.json'), 'w') as f:
    annotations = json.dump(annotations, f)
    f.close()