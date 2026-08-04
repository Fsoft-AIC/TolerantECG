CUDA_VISIBLE_DEVICES=0,1,2,3 \
python src/train.py \
\
trainer=deepspeed \
trainer.devices=4 \
trainer.max_epochs=10 \
trainer.precision="bf16" \
\
data=mimic_dino_mask_noise \
data.sample_rate=500 \
data.denoising=False \
data.is_rag=True \
data.subset_percent=1.0 \
data.num_workers=8 \
data.batch_size=256 \
data.local_num=8 \
data.global_num=2 \
data.noise_probs="[0.7,0.7,0.7]" \
data.noise_rate=0.5 \
data.snr_db="[-10,0]" \
data.global_mask_scale="[6,12]" \
data.local_mask_scale="[1,6]" \
\
model=dino_clip \
model/loss_function=clip \
model.alpha=1.0 \
model.beta=1.0 \
model.out_dim=768 \
\
model.text_encoder.is_finetune=False \
\
model.ecg_encoder.in_chans=12 \
\
callbacks=dino_clip