CUDA_VISIBLE_DEVICES=0 \
\
python src/train.py \
\
trainer=deepspeed \
trainer.devices=1 \
trainer.max_epochs=10 \
trainer.precision="bf16" \
\
data=ptb_xl_compose \
data.sample_rate=500 \
data.denoising=False \
data.snr_db=[-10,0] \
data.batch_size=16 \
data.num_workers=4 \
data.level="diagnostic_super" \
\
model=dino_clip_finetune \
model.ecg_encoder.in_chans=12 \
model.ecg_encoder.num_dim=1 \
model.ecg_encoder.head_bias=False \
model.ecg_encoder.last_norm=True \
\
model/loss_function=bce \
model.is_finetune=True \
model.optimizer.lr=5e-6 \
model.ckpt_path="path/to/encoder.pt" \
\
callbacks=ptb_xl \
~callbacks.wandb_callback \
model.num_classes=5 \