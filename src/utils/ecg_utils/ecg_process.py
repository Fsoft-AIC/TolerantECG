from typing import Union
import random
import numpy as np
import torch
from scipy.signal import resample


def calculate_mask_lead(ecg_signal):
    """
    ecg: (12, 5000)

    Calculate missing limb leads knowing 2 pairs
    """
    if isinstance(ecg_signal, np.ndarray):
        re_ecg = np.copy(ecg_signal)
    if isinstance(ecg_signal, torch.Tensor):
        re_ecg = ecg_signal.clone()

    # Order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6

    # lead I
    if (re_ecg[0] == 0).all() and (re_ecg[1] != 0).any() and (re_ecg[2] != 0).any():
        re_ecg[0] = re_ecg[1] - re_ecg[2]   # I = II - III
    if (re_ecg[0] == 0).all() and (re_ecg[3] != 0).any() and (re_ecg[4] != 0).any():
        re_ecg[0] = 2.0 / 3.0 * (re_ecg[4] - re_ecg[3])   # I = 2/3 * (avL - avR)

    # lead II
    if (re_ecg[1] == 0).all() and (re_ecg[0] != 0).any() and (re_ecg[2] != 0).any():
        re_ecg[1] = re_ecg[0] - re_ecg[2]   # II = I + III
    if (re_ecg[1] == 0).all() and (re_ecg[5] != 0).any() and (re_ecg[3] != 0).any():
        re_ecg[1] = 2.0 / 3.0 * (re_ecg[5] - re_ecg[3])   # II = 2/3 * (avF - avR)

    # lead III
    if (re_ecg[2] == 0).all() and (re_ecg[0] != 0).any() and (re_ecg[1] != 0).any():
        re_ecg[2] = re_ecg[1] - re_ecg[0]   # III = II - I
    if (re_ecg[2] == 0).all() and (re_ecg[5] != 0).any() and (re_ecg[4] != 0).any():
        re_ecg[2] = 2.0 / 3.0 * (re_ecg[5] - re_ecg[4])   # III = 2/3 * (avF - avL)

    # lead avR
    if (re_ecg[3] == 0).all() and (re_ecg[0] != 0).any() and (re_ecg[1] != 0).any():
        re_ecg[3] = -1.0 / 2.0 * (re_ecg[0] + re_ecg[1])   # avR = -1/2 * (I + II)

    # lead avL
    if (re_ecg[4] == 0).all() and (re_ecg[0] != 0).any() and (re_ecg[2] != 0).any():
        re_ecg[4] = 1.0 / 2.0 * (re_ecg[0] - re_ecg[2])

    # lead avF
    if (re_ecg[5] == 0).all() and (re_ecg[1] != 0).any() and (re_ecg[2] != 0).any():
        re_ecg[5] = 1.0 / 2.0 * (re_ecg[1] + re_ecg[2])

    return re_ecg

def difficult_mask(ecg, difficulty='easy', seed=None):
    assert difficulty in ['easy', 'medium', 'hard']
    if difficulty == "easy":
        return random_percent_keep(ecg, p=(0.75, 1.0), seed=seed)   # 9->12
    if difficulty == "medium":
        return random_percent_keep(ecg, p=(0.42, 0.67), seed=seed)  # 5->8
    if difficulty == 'hard':
        return random_percent_keep(ecg, p=(0.084, 0.34), seed=seed) # 1->4

def random_percent_keep(tensor, p = (0.5, 1.0), seed=None):
    random.seed(seed)  # Set the seed for Python's random module

    # Randomly select a percentage between p1 and p2
    if not isinstance(p, (float, int)):
        p1, p2 = p
        p = random.uniform(p1, p2)

    # Calculate the number of channels
    num_channels = tensor.size(0)

    # Calculate the number of channels to keep
    num_channels_to_keep = int(num_channels * p)

    # Randomly select the indices of the channels to keep
    keep_indices = sorted(random.sample(range(num_channels), num_channels_to_keep))

    # Create a mask with ones for the channels to keep and zeros for others
    mask = torch.zeros_like(tensor)
    mask[keep_indices] = 1

    # Apply the mask to the tensor
    masked_tensor = tensor * mask

    return masked_tensor

def random_lead_keep(tensor, p = (1, 11), seed=None):
    random.seed(seed)  # Set the seed for Python's random module

    # Randomly select a percentage between p1 and p2
    if not isinstance(p, (float, int)):
        p1, p2 = p
        p = random.randint(p1, p2)

    # Calculate the number of channels
    num_channels = tensor.size(0)

    # Calculate the number of channels to keep
    num_channels_to_keep = int(p)

    # Randomly select the indices of the channels to keep
    keep_indices = sorted(random.sample(range(num_channels), num_channels_to_keep))

    # Create a mask with ones for the channels to keep and zeros for others
    mask = torch.zeros_like(tensor)
    mask[keep_indices] = 1

    # Apply the mask to the tensor
    masked_tensor = tensor * mask

    return masked_tensor, mask[:, 0].to(torch.float32)

def keep_lead(ecg_signal, p=[0, 1, 2, 3, 4, 5], **kwargs):
    lead_mask = torch.zeros(ecg_signal.shape[0])
    lead_mask[p] = 1

    masked_ecg = ecg_signal * lead_mask.unsqueeze(1)

    return masked_ecg, lead_mask.to(torch.float32)

def mask_ecg_real(ecg_signal, num_leads=None, seed=None):
    """
    ecg: (12, 5000)

    Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    Six leads: I, II, III, aVR, aVL, aVF
    Five leads: I, II, III, avR, avF
    Three leads: I, II and III
    One leads: Any
    """
    assert num_leads in [None, 1, 3, 4, 6, 12]
    np.random.seed(seed)
    random.seed(seed)

    if isinstance(ecg_signal, torch.Tensor):
        masked_signal = ecg_signal.clone()

    if isinstance(ecg_signal, np.ndarray):
        masked_signal = np.copy(ecg_signal)

    lead_group = np.array([np.array([0, 1, 2]), np.array([3, 4, 5]), np.random.choice([6, 7, 8, 9, 10, 11], 3, replace=False)])

    # keeping leads
    masks = {
        12: np.arange(12),  # All leads
        6: lead_group[np.random.choice(3, 2, replace=False)].flatten(),
        4: np.concatenate([np.array([0, 1, 2]), np.random.choice([6, 7, 8, 9, 10, 11], 1, replace=False)]),
        3: np.array([0, 1, 2]),
        1: np.array(np.random.choice(12, 1, replace=False))
    }

    if num_leads is None:
        num_leads = random.randint(0, 3)
        num_leads = list(masks.keys())[num_leads]
    # Get the mask for the specified number of leads
    mask = np.sort(masks.get(num_leads))

    # Set the values of the not chosen leads to zero
    all_indices = set(range(12))
    not_chosen_indices = all_indices - set(mask)
    masked_signal[list(not_chosen_indices)] = 0.0
    
    return masked_signal


def mask_ecg_doctor(ecg_signal, seed=None):
    """
    Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    Six leads: (I, II, III), (aVR, aVL, aVF), (3 in V1-V6) choose 2 groups
    Five leads: I, II, III, (2 in aVs and Vs)
    Four leads: I, II, III, V
    Three leads: (I, II, III), (aVR, aVL, aVF), (3 in V1-V6)  choose 1 group
    # Two leads: I, II
    One leads: Any
    """
    np.random.seed(seed)
    random.seed(seed)

    is_torch = False
    if isinstance(ecg_signal, torch.Tensor):
        is_torch = True
        ecg_signal = ecg_signal.numpy()

    lead_group = np.array([np.array([0, 1, 2]), np.array([3, 4, 5]), np.random.choice([6, 7, 8, 9, 10, 11], 3, replace=False)])

    masks = {
        12: np.arange(12),  # All leads
        6: lead_group[np.random.choice(3, 2, replace=False)].flatten(),
        5: np.concatenate([np.array([0, 1, 2]), np.random.choice([3, 4, 5, 6, 7, 8, 9, 10, 11], 2, replace=False)]),
        4: np.array([0, 1, 2, np.random.choice([6, 7, 8, 9, 10, 11])]),
        3: lead_group[np.random.choice(3, 1, replace=False)].flatten(),
        1: np.array([np.random.choice(12)])
    }

    num_leads = random.randint(0, 5)    # 0 -> 5
    num_leads = list(masks.keys())[num_leads]
    # Get the mask for the specified number of leads
    mask = np.sort(masks.get(num_leads))

    if mask is None:
        raise ValueError("Invalid number of leads specified.")

    # Apply the mask to the ECG signal
    masked_signal = ecg_signal.copy()

    # Set the values of the not chosen leads to zero
    all_indices = set(range(12))
    not_chosen_indices = all_indices - set(mask)
    masked_signal[list(not_chosen_indices)] = 0.0

    if is_torch:
        masked_signal = torch.from_numpy(masked_signal)
    return masked_signal


def mask_ecg(ecg, p: Union[float, list, tuple], seed=None):
    """
    ecg: (12, 5000)
    """
    np.random.seed(seed)

    if isinstance(ecg, torch.Tensor):
        masked_ecg = ecg.clone()

    if isinstance(ecg, np.ndarray):
        masked_ecg = np.copy(ecg)
    # Get the number of channels
    num_leads = ecg.shape[0]

    # Generate a mask with probability p
    mask = np.random.rand(num_leads) < p
    
    # Ensure at least one channel is unmasked
    if mask.all():
        # Randomly select one channel to unmask
        unmask_index = np.random.randint(num_leads)
        mask[unmask_index] = False
    
    # Apply the mask to the array
    masked_ecg[mask] = 0.0  # Assuming masking means setting to 0

    return masked_ecg, torch.from_numpy(~mask).to(torch.float32)


def norm_ecg(ecg):
    """
    ecg: (Num_lead, Seq_len)
    """
    min_value = ecg.min(dim=1, keepdim=True)[0]
    max_value = ecg.max(dim=1, keepdim=True)[0]
    norm_ecg = (ecg - min_value) / (max_value - min_value)

    return norm_ecg


def mean_ecg(ecg):
    """
    ecg: (12, 5000)
    """
    mean = torch.mean(ecg, dim=1, keepdim=True)
    return ecg - mean


def resample_signal(signal, old_sr=360, new_sr=500):
    seq_len = signal.shape[-1]
    resampled_signal = resample(signal, int(seq_len * new_sr / old_sr), axis=-1)

    return resampled_signal