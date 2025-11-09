"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations
import comfy.utils
import comfy.model_management
import logging
import torch

LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def model_lora_keys_clip(model, key_map={}):
    sdk = model.state_dict().keys()
    for k in sdk:
        if k.endswith(".weight"):
            key_map["text_encoders.{}".format(k[:-len(".weight")])] = k #generic lora format without any weird key names

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present = False
    clip_g_present = False
    for b in range(32): #TODO: clean up
        for c in LORA_CLIP_MAP:
            k = "clip_h.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                key_map[lora_key] = k

            k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                key_map[lora_key] = k
                clip_l_present = True
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                key_map[lora_key] = k

            k = "clip_g.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                clip_g_present = True
                if clip_l_present:
                    lora_key = "lora_te2_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                    key_map[lora_key] = k
                    lora_key = "text_encoder_2.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k
                else:
                    lora_key = "lora_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #TODO: test if this is correct for SDXL-Refiner
                    key_map[lora_key] = k
                    lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k
                    lora_key = "lora_prior_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #cascade lora: TODO put lora key prefix in the model config
                    key_map[lora_key] = k

    for k in sdk:
        if k.endswith(".weight"):
            if k.startswith("t5xxl.transformer."):#OneTrainer SD3 and Flux lora
                l_key = k[len("t5xxl.transformer."):-len(".weight")]
                t5_index = 1
                if clip_g_present:
                    t5_index += 1
                if clip_l_present:
                    t5_index += 1
                    if t5_index == 2:
                        key_map["lora_te{}_{}".format(t5_index, l_key.replace(".", "_"))] = k #OneTrainer Flux
                        t5_index += 1

                key_map["lora_te{}_{}".format(t5_index, l_key.replace(".", "_"))] = k
            elif k.startswith("hydit_clip.transformer.bert."): #HunyuanDiT Lora
                l_key = k[len("hydit_clip.transformer.bert."):-len(".weight")]
                lora_key = "lora_te1_{}".format(l_key.replace(".", "_"))
                key_map[lora_key] = k


    k = "clip_g.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_prior_te_text_projection"] = k #cascade lora?
        # key_map["text_encoder.text_projection"] = k #TODO: check if other lora have the text_projection too
        key_map["lora_te2_text_projection"] = k #OneTrainer SD3 lora

    k = "clip_l.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_te1_text_projection"] = k #OneTrainer SD3 lora, not necessary but omits warning

    return key_map

def model_lora_keys_unet(model, key_map={}):
    sd = model.state_dict()
    sdk = sd.keys()

    for k in sdk:
        if k.startswith("diffusion_model."):
            if k.endswith(".weight"):
                key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = k
                key_map["{}".format(k[:-len(".weight")])] = k #generic lora format without any weird key names
            else:
                key_map["{}".format(k)] = k #generic lora format for not .weight without any weird key names

    diffusers_keys = comfy.utils.unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[:-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key
            key_map["lycoris_{}".format(key_lora)] = unet_key #simpletuner lycoris format

            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(p, k[:-len(".weight")].replace(".to_", ".processor.to_"))
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key

    if isinstance(model, comfy.model_base.StableCascade_C):
        for k in sdk:
            if k.startswith("diffusion_model."):
                if k.endswith(".weight"):
                    key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
                    key_map["lora_prior_unet_{}".format(key_lora)] = k

    if isinstance(model, comfy.model_base.SD3): #Diffusers lora SD3
        diffusers_keys = comfy.utils.mmdit_to_diffusers(model.model_config.unet_config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_lora = "transformer.{}".format(k[:-len(".weight")]) #regular diffusers sd3 lora format
                key_map[key_lora] = to

                key_lora = "base_model.model.{}".format(k[:-len(".weight")]) #format for flash-sd3 lora and others?
                key_map[key_lora] = to

                key_lora = "lora_transformer_{}".format(k[:-len(".weight")].replace(".", "_")) #OneTrainer lora
                key_map[key_lora] = to

                key_lora = "lycoris_{}".format(k[:-len(".weight")].replace(".", "_")) #simpletuner lycoris format
                key_map[key_lora] = to

    if isinstance(model, comfy.model_base.AuraFlow): #Diffusers lora AuraFlow
        diffusers_keys = comfy.utils.auraflow_to_diffusers(model.model_config.unet_config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_lora = "transformer.{}".format(k[:-len(".weight")]) #simpletrainer and probably regular diffusers lora format
                key_map[key_lora] = to

    if isinstance(model, comfy.model_base.PixArt):
        diffusers_keys = comfy.utils.pixart_to_diffusers(model.model_config.unet_config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_lora = "transformer.{}".format(k[:-len(".weight")]) #default format
                key_map[key_lora] = to

                key_lora = "base_model.model.{}".format(k[:-len(".weight")]) #diffusers training script
                key_map[key_lora] = to

                key_lora = "unet.base_model.model.{}".format(k[:-len(".weight")]) #old reference peft script
                key_map[key_lora] = to

    if isinstance(model, comfy.model_base.HunyuanDiT):
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"):
                key_lora = k[len("diffusion_model."):-len(".weight")]
                key_map["base_model.model.{}".format(key_lora)] = k #official hunyuan lora format

    if isinstance(model, comfy.model_base.Flux): #Diffusers lora Flux
        diffusers_keys = comfy.utils.flux_to_diffusers(model.model_config.unet_config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_map["transformer.{}".format(k[:-len(".weight")])] = to #simpletrainer and probably regular diffusers flux lora format
                key_map["lycoris_{}".format(k[:-len(".weight")].replace(".", "_"))] = to #simpletrainer lycoris
                key_map["lora_transformer_{}".format(k[:-len(".weight")].replace(".", "_"))] = to #onetrainer
        for k in sdk:
            hidden_size = model.model_config.unet_config.get("hidden_size", 0)
            if k.endswith(".weight") and ".linear1." in k:
                key_map["{}".format(k.replace(".linear1.weight", ".linear1_qkv"))] = (k, (0, 0, hidden_size * 3))

    if isinstance(model, comfy.model_base.GenmoMochi):
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"): #Official Mochi lora format
                key_lora = k[len("diffusion_model."):-len(".weight")]
                key_map["{}".format(key_lora)] = k

    if isinstance(model, comfy.model_base.HunyuanVideo):
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"):
                # diffusion-pipe lora format
                key_lora = k
                key_lora = key_lora.replace("_mod.lin.", "_mod.linear.").replace("_attn.qkv.", "_attn_qkv.").replace("_attn.proj.", "_attn_proj.")
                key_lora = key_lora.replace("mlp.0.", "mlp.fc1.").replace("mlp.2.", "mlp.fc2.")
                key_lora = key_lora.replace(".modulation.lin.", ".modulation.linear.")
                key_lora = key_lora[len("diffusion_model."):-len(".weight")]
                key_map["transformer.{}".format(key_lora)] = k
                key_map["diffusion_model.{}".format(key_lora)] = k  # Old loras

    if isinstance(model, comfy.model_base.HiDream):
        for k in sdk:
            if k.startswith("diffusion_model."):
                if k.endswith(".weight"):
                    key_lora = k[len("diffusion_model."):-len(".weight")]
                    key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = k #SimpleTuner lycoris format
                    key_map["transformer.{}".format(key_lora)] = k #SimpleTuner regular format

    if isinstance(model, comfy.model_base.ACEStep):
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"): #Official ACE step lora format
                key_lora = k[len("diffusion_model."):-len(".weight")]
                key_map["{}".format(key_lora)] = k

    if isinstance(model, comfy.model_base.Omnigen2):
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"):
                key_lora = k[len("diffusion_model."):-len(".weight")]
                key_map["{}".format(key_lora)] = k

    if isinstance(model, comfy.model_base.QwenImage):
        for k in sdk:
            if k.startswith("diffusion_model.") and k.endswith(".weight"): #QwenImage lora format
                key_lora = k[len("diffusion_model."):-len(".weight")]
                # Direct mapping for transformer_blocks format (QwenImage LoRA format)
                key_map["{}".format(key_lora)] = k
                # Support transformer prefix format
                key_map["transformer.{}".format(key_lora)] = k
                key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = k #SimpleTuner lycoris format

    return key_map


def pad_tensor_to_shape(tensor: torch.Tensor, new_shape: list[int]) -> torch.Tensor:
    """
    Pad a tensor to a new shape with zeros.

    Args:
        tensor (torch.Tensor): The original tensor to be padded.
        new_shape (List[int]): The desired shape of the padded tensor.

    Returns:
        torch.Tensor: A new tensor padded with zeros to the specified shape.

    Note:
        If the new shape is smaller than the original tensor in any dimension,
        the original tensor will be truncated in that dimension.
    """
    if any([new_shape[i] < tensor.shape[i] for i in range(len(new_shape))]):
        raise ValueError("The new shape must be larger than the original tensor in all dimensions")

    if len(new_shape) != len(tensor.shape):
        raise ValueError("The new shape must have the same number of dimensions as the original tensor")

    # Create a new tensor filled with zeros
    padded_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

    # Create slicing tuples for both tensors
    orig_slices = tuple(slice(0, dim) for dim in tensor.shape)
    new_slices = tuple(slice(0, dim) for dim in tensor.shape)

    # Copy the original tensor into the new tensor
    padded_tensor[new_slices] = tensor[orig_slices]

    return padded_tensor

def calculate_weight(patches, weight, key, intermediate_dtype=torch.float32, original_weights=None):
    for p in patches:
        strength = p[0]
        v = p[1]
        strength_model = p[2]
        offset = p[3]
        function = p[4]
        if function is None:
            function = lambda a: a

        old_weight = None
        if offset is not None:
            old_weight = weight
            weight = weight.narrow(offset[0], offset[1], offset[2])

        if strength_model != 1.0:
            weight *= strength_model

        if isinstance(v, list):
            v = (calculate_weight(v[1:], v[0][1](comfy.model_management.cast_to_device(v[0][0], weight.device, intermediate_dtype, copy=True), inplace=True), key, intermediate_dtype=intermediate_dtype), )


        if len(v) == 1:
            patch_type = "diff"
        elif len(v) == 2:
            patch_type = v[0]
            v = v[1]

        if patch_type == "diff":
            diff: torch.Tensor = v[0]
            # An extra flag to pad the weight if the diff's shape is larger than the weight
            do_pad_weight = len(v) > 1 and v[1]['pad_weight']
            if do_pad_weight and diff.shape != weight.shape:
                logging.info("Pad weight {} from {} to shape: {}".format(key, weight.shape, diff.shape))
                weight = pad_tensor_to_shape(weight, diff.shape)

            if strength != 0.0:
                if diff.shape != weight.shape:
                    logging.warning("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, diff.shape, weight.shape))
                else:
                    weight += function(strength * comfy.model_management.cast_to_device(diff, weight.device, weight.dtype))
        elif patch_type == "set":
            weight.copy_(v[0])
        elif patch_type == "model_as_lora":
            target_weight: torch.Tensor = v[0]
            diff_weight = comfy.model_management.cast_to_device(target_weight, weight.device, intermediate_dtype) - \
                          comfy.model_management.cast_to_device(original_weights[key][0][0], weight.device, intermediate_dtype)
            weight += function(strength * comfy.model_management.cast_to_device(diff_weight, weight.device, weight.dtype))
        else:
            logging.warning("patch type not recognized {} {}".format(patch_type, key))

        if old_weight is not None:
            weight = old_weight

    return weight
