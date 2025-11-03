"""
Latent format for Wan2.2 model
Handles latent space normalization and denormalization
"""

import torch


class Wan22LatentFormat:
    """
    Wan2.2 latent format with 48 channels
    Handles normalization using mean and std statistics
    """
    
    latent_channels = 48
    latent_dimensions = 3
    
    latent_rgb_factors = [
        [ 0.0119,  0.0103,  0.0046],
        [-0.1062, -0.0504,  0.0165],
        [ 0.0140,  0.0409,  0.0491],
        [-0.0813, -0.0677,  0.0607],
        [ 0.0656,  0.0851,  0.0808],
        [ 0.0264,  0.0463,  0.0912],
        [ 0.0295,  0.0326,  0.0590],
        [-0.0244, -0.0270,  0.0025],
        [ 0.0443, -0.0102,  0.0288],
        [-0.0465, -0.0090, -0.0205],
        [ 0.0359,  0.0236,  0.0082],
        [-0.0776,  0.0854,  0.1048],
        [ 0.0564,  0.0264,  0.0561],
        [ 0.0006,  0.0594,  0.0418],
        [-0.0319, -0.0542, -0.0637],
        [-0.0268,  0.0024,  0.0260],
        [ 0.0539,  0.0265,  0.0358],
        [-0.0359, -0.0312, -0.0287],
        [-0.0285, -0.1032, -0.1237],
        [ 0.1041,  0.0537,  0.0622],
        [-0.0086, -0.0374, -0.0051],
        [ 0.0390,  0.0670,  0.2863],
        [ 0.0069,  0.0144,  0.0082],
        [ 0.0006, -0.0167,  0.0079],
        [ 0.0313, -0.0574, -0.0232],
        [-0.1454, -0.0902, -0.0481],
        [ 0.0714,  0.0827,  0.0447],
        [-0.0304, -0.0574, -0.0196],
        [ 0.0401,  0.0384,  0.0204],
        [-0.0758, -0.0297, -0.0014],
        [ 0.0568,  0.1307,  0.1372],
        [-0.0055, -0.0310, -0.0380],
        [ 0.0239, -0.0305,  0.0325],
        [-0.0663, -0.0673, -0.0140],
        [-0.0416, -0.0047, -0.0023],
        [ 0.0166,  0.0112, -0.0093],
        [-0.0211,  0.0011,  0.0331],
        [ 0.1833,  0.1466,  0.2250],
        [-0.0368,  0.0370,  0.0295],
        [-0.3441, -0.3543, -0.2008],
        [-0.0479, -0.0489, -0.0420],
        [-0.0660, -0.0153,  0.0800],
        [-0.0101,  0.0068,  0.0156],
        [-0.0690, -0.0452, -0.0927],
        [-0.0145,  0.0041,  0.0015],
        [ 0.0421,  0.0451,  0.0373],
        [ 0.0504, -0.0483, -0.0356],
        [-0.0837,  0.0168,  0.0055]
    ]
    
    latent_rgb_factors_bias = [0.0317, -0.0878, -0.1388]
    
    def __init__(self):
        self.scale_factor = 1.0
        self.latents_mean = torch.tensor([
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
            -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
            -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
            0.0124, 0.0458, -0.0926, 0.1520, 0.0395, 0.1149, -0.1107, 0.1566,
            -0.1869, -0.0732, 0.0711, -0.1868, -0.2065, -0.2284, 0.0168, -0.0013
        ]).view(1, self.latent_channels, 1, 1, 1)
        
        self.latents_std = torch.tensor([
            3.4005, 2.0118, 2.5788, 2.7188, 1.8340, 2.0842, 2.7137, 2.5093,
            3.6517, 2.8230, 3.0520, 1.9479, 2.2155, 1.6550, 3.2140, 2.4261,
            2.8918, 2.9863, 2.4314, 2.8265, 2.4602, 2.5803, 2.6160, 2.2869,
            2.8898, 2.5972, 2.8829, 2.5548, 2.5403, 2.7629, 2.5956, 2.5023,
            2.7291, 2.4441, 2.9670, 2.7357, 2.9390, 2.9180, 2.9052, 2.7060,
            2.9077, 2.9449, 2.6929, 2.9099, 2.9088, 3.1041, 3.0048, 3.0033
        ]).view(1, self.latent_channels, 1, 1, 1)

    def process_in(self, latent):
        """
        Normalize latent tensor for model input
        
        Args:
            latent: Raw latent tensor
            
        Returns:
            Normalized latent tensor
        """
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        """
        Denormalize latent tensor from model output
        
        Args:
            latent: Normalized latent tensor
            
        Returns:
            Denormalized latent tensor
        """
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean
