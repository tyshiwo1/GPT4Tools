CUDA_VISIBLE_DEVICES=3 python gpt4tools.py --base_model "lmsys/vicuna-13b-v1.5" --lora_model "./output/gpt4tools/checkpoint-400" --llm_device "cpu" --load "MagicBrush_cuda:0,ImageCaptioning_cuda:0"

CUDA_VISIBLE_DEVICES=0 python gpt4tools.py --base_model "lmsys/vicuna-13b-v1.5" --lora_model "stevengrove/gpt4tools-vicuna-13b-lora" --llm_device "cpu" --load "MagicBrush_cuda:0,ImageCaptioning_cuda:0"

ssh -p 22000 -N -f -L 127.0.0.1:2345:127.0.0.1:2345 yaoteng@147.8.183.113

NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch  --nproc_per_node=3 --master_port=29005 lora_finetune.py  --base_model "lmsys/vicuna-13b-v1.5"  --data_path "annotations_path.json"  --output_dir output/gpt4tools  --prompt_template_name gpt4tools  --num_epochs 6  --batch_size 3  --cutoff_len 2048  --group_by_length  --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]'  --lora_r 16  --micro_batch_size=1 

NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch  --nproc_per_node=3 --master_port=29005 lora_finetune.py  --base_model "lmsys/vicuna-13b-v1.5" --lora_model "stevengrove/gpt4tools-vicuna-13b-lora"  --data_path "annotations_path.json"  --output_dir output/gpt4tools  --prompt_template_name gpt4tools  --num_epochs 6  --batch_size 3  --cutoff_len 2048  --group_by_length  --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]'  --lora_r 16  --micro_batch_size=1 

NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch  --nproc_per_node=3 --master_port=29005 lora_finetune.py  --base_model "lmsys/vicuna-13b-v1.5" --resume_from_checkpoint "stevengrove/gpt4tools-vicuna-13b-lora"  --data_path "annotations_path.json"  --output_dir output/gpt4tools  --prompt_template_name gpt4tools  --num_epochs 6  --batch_size 3  --cutoff_len 2048  --group_by_length  --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]'  --lora_r 16  --micro_batch_size=1 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=10027 tools/train.py ./configs/lobster/lobster_r50_1x_coco_splitp_more_unequ_stage_shareweight_lighttail_nonll_alpha_bgfg2fg_iou_upperbound_stage23456.py --launcher pytorch --resume-from ./work_dirs/lobster_ori_focal_0821_1118_test/epoch_8.pth


scp -r -P 22000 yaoteng@147.8.183.113:/home/yaoteng/checkpoints/





# data preparation

python3 get_cap_magicbrush.py --dev "dev" --global-descriptions "global_descriptions.json" --local-descriptions "local_descriptions.json" --caption-path "devtest_magicbrash_caption.json"
python3 get_cap_magicbrush.py --dev "test" --global-descriptions "global_descriptions.json" --local-descriptions "local_descriptions.json" --caption-path "devtest_magicbrash_caption.json"
python3 scripts/get_instruction_magicbrush.py --caption-path "devtest_magicbrash_caption.json" --instruction-path "devtest_magicbrash_instruction.json"
python3 scripts/get_instruction_magicbrush.py --caption-path "devtest_magicbrash_caption.json" --instruction-path "gpt4tools_71k.json"