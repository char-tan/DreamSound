import os
import math
import numbers
import librosa
import librosa.display
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def standardise(spec, mean, std):
    return (spec - mean) / std

def destandardise(spec, mean, std):
    return (spec * std) + mean

def wav_to_mel_dB(wav_tensor, sample_rate=22050, n_fft=441, hop_length=200, n_mels=128):
    mel = librosa.feature.melspectrogram(wav_tensor, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_dB = librosa.power_to_db(mel)

    return mel_dB

def mel_dB_to_wav(mel_dB, sample_rate=22050, n_fft=441, hop_length=200, n_mels=128):
    mel = librosa.db_to_power(mel_dB)
    wav = librosa.feature.inverse.mel_to_audio(mel, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)

    return wav

def plot_mel_dB(mel_dB, fig, ax, ax_index):
    img = librosa.display.specshow(mel_dB, x_axis='time', y_axis='mel', sr=22050, ax=ax[ax_index])
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

def plot_wav(wav, fig, ax, ax_index):
    librosa.display.waveshow(wav, 22050, ax=ax[ax_index])

def get_new_shape(opt, base_shape, pyramid_level):
    "finds the shape of the pyramid level"

    input_shape = base_shape[-2:]

    SHAPE_MARGIN = 8
    pyramid_ratio = opt.pyramid_ratio
    pyramid_size = opt.pyramid_size
    exponent = pyramid_level - pyramid_size + 1
    new_shape = np.round(np.float32(input_shape)*(pyramid_ratio**exponent)).astype(np.int32)

    if new_shape[0] < SHAPE_MARGIN or new_shape[1] < SHAPE_MARGIN:
        print(f'Pyramid size {opt.pyramid_size} with pyramid ratio {opt.pyramid_ratio} gives too small pyramid levels with size={new_shape}')
        print(f'Please change parameters.')
        exit(0)

    return new_shape

def get_pyramid_roll_max(opt, base_shape, pyramid_level):
    "finds the max allowed rolling for the pyramid level"

    pyramid_ratio = opt.pyramid_ratio
    pyramid_size = opt.pyramid_size
    exponent = pyramid_level - pyramid_size + 1

    return np.around(opt.pyramid_roll * pyramid_ratio ** exponent).astype(np.int32)

def ouroboros_transform(opt, wav_in, output_length):
    "converts to wav, stretchs and crops, analogous to zooming / rotation of DeepDream ouroboros videos"

    if opt.ouroboros_transform == "time_stretch":
        transformed = librosa.effects.time_stretch(wav_in, opt.ouroboros_stretch) 

    elif opt.ouroboros_transform == "resample":
        transformed = librosa.resample(wav_in, 22050, 22050 + opt.ouroboros_sr_offset)
    
    wav_out = transformed[-output_length:]

    return wav_out

class CascadeGaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing separately for each channel (depthwise convolution).

    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.

    """
    def __init__(self, kernel_size, sigma):
        super().__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]  # std multipliers
        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]  # isotropic Gaussian

        self.pad = int(kernel_size[0] / 2)  # assure we have the same spatial resolution

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for sigma in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * torch.exp(-((grid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            # Normalize - make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)
            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            #kernel = kernel.repeat(3, 1, 1, 1)
            kernel = kernel.cuda()

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = F.conv2d

    def forward(self, input):
        input = F.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')

        # Apply Gaussian kernels depthwise over the input (hence groups equals the number of input channels)
        # shape = (1, 3, H, W) -> (1, 3, H, W)
        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3

