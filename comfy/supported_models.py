import torch
from . import model_base
import comfy.text_encoders.wan
from . import supported_models_base
from . import latent_formats


class WAN21_T2V(supported_models_base.BASE):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "t2v",
    }

    sampling_settings = {
        "shift": 8.0,
    }

    unet_extra_config = {}
    latent_format = latent_formats.Wan21

    memory_usage_factor = 0.9

    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = self.unet_config.get("dim", 2000) / 2222

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}umt5xxl.transformer.".format(pref))
        return supported_models_base.ClipTarget(comfy.text_encoders.wan.WanT5Tokenizer, comfy.text_encoders.wan.te(**t5_detect))

class WAN21_I2V(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "i2v",
        "in_dim": 36,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21(self, image_to_video=True, device=device)
        return out

class WAN21_FunControl2V(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "i2v",
        "in_dim": 48,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21(self, image_to_video=False, device=device)
        return out

class WAN21_Camera(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "camera",
        "in_dim": 32,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21_Camera(self, image_to_video=False, device=device)
        return out

class WAN22_Camera(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "camera_2.2",
        "in_dim": 36,
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21_Camera(self, image_to_video=False, device=device)
        return out

class WAN21_Vace(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "vace",
    }

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = 1.2 * self.memory_usage_factor

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21_Vace(self, image_to_video=False, device=device)
        return out

class WAN21_HuMo(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "humo",
    }

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN21_HuMo(self, image_to_video=False, device=device)
        return out

class WAN22_S2V(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "s2v",
    }

    def __init__(self, unet_config):
        super().__init__(unet_config)

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN22_S2V(self, device=device)
        return out

class WAN22_Animate(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "animate",
    }

    def __init__(self, unet_config):
        super().__init__(unet_config)

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN22_Animate(self, device=device)
        return out

class WAN22_T2V(WAN21_T2V):
    unet_config = {
        "image_model": "wan2.1",
        "model_type": "t2v",
        "out_dim": 48,
    }

    latent_format = latent_formats.Wan22

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.WAN22(self, image_to_video=True, device=device)
        return out

models = [WAN22_T2V, WAN21_T2V, WAN21_I2V, WAN21_FunControl2V, WAN21_Vace, WAN21_Camera, WAN22_Camera, WAN22_S2V, WAN21_HuMo, WAN22_Animate]
