# pip install accelerate
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def get_blip(
    blip_name="Salesforce/blip2-flan-t5-xl", 
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./checkpoints/models",
):
    processor = Blip2Processor.from_pretrained(
        blip_name,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        blip_name, 
        torch_dtype=torch_dtype, 
        device_map=device_map,
    )
    return processor, model

def get_blip_res(inputs_pil, processor, model, device, dtype, question="What exactly does this picture describe?"):
    inputs = processor(inputs_pil, question, return_tensors="pt").to(device, dtype)
    out = model.generate(**inputs)
    res = processor.decode(out[0], skip_special_tokens=True)
    del inputs
    del out
    return res
    

if __name__ == "__main__":
    #CUDA_VISIBLE_DEVICES=3 python ./image_caption/image_caption.py
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    
    processor, model = get_blip()
    out = get_blip_res(raw_image, processor, model, device="cuda", dtype=torch.float16)
    
    print(processor.decode(out[0], skip_special_tokens=True))
