CUDA_VISIBLE_DEVICES=0 python gpt4tools.py --base_model "lmsys/vicuna-13b-v1.5" --lora_model "stevengrove/gpt4tools-vicuna-13b-lora" --llm_device "cuda" --load "InstructPix2Pix_cuda:0,ImageCaptioning_cuda:0"

ssh -p 22000 -N -f -L 127.0.0.1:2345:127.0.0.1:2345 yaoteng@147.8.183.113