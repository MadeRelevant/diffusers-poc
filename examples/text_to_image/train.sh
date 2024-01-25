#/usr/bin/bash

accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=SG161222/RealVisXL_V3.0 \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --dataloader_num_workers=0 \
  --resolution=1024 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --train_data_dir=./data \
  --num_train_epochs=10 \
  --max_train_steps=100 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=output/ \
  --validation_epochs=20 \
  --validation_prompt="RAW photo, aifluencer, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3" \
  --train_text_encoder\
  --checkpointing_steps=50 \
  --seed=1337