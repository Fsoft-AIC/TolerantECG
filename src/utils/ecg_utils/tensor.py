import gc
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

def denormalize_image(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
    ten = x.clone()     # C, H, W 
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(ten, 0, 1)

def denormalize_batch(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
    # B, C, H, W -> C, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # C, H, W, B -> B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def clean_memory(*args):
    for var in args:
        del var

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def half_image(image: np.array):
    original_height, original_width = image.shape[:2]  # OpenCV
    new_width = original_width // 2
    new_height = original_height // 2

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)  # OpenCV

    clean_memory(original_height, original_width, new_height, new_width)

    return resized_image

def draw_spectrogram(ecg, n_fft, sample_rate):
    """
    ecg: [Freq, Time]
    """
    # 59 -> 0
    y_values = [round(k * sample_rate / n_fft, 2) for k in range(len(ecg) - 1, -1, -1)]
    plt.imshow(ecg)
    plt.gca().set_yticklabels(y_values)
    plt.ylabel("Hz", fontsize=12)
    plt.xlabel("Frame", fontsize=12)