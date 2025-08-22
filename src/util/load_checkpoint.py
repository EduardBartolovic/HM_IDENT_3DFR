import os

import torch

from src.util.misc import colorstr


def load_rgbd_backbone_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    # Splitting the checkpoint for RGB and Depth Inputs
    rgb_checkpoint = {}
    depth_checkpoint = {}

    for k, v in checkpoint.items():
        if 'rgb' in k:
            new_key = k.replace('rgb_', '')
            rgb_checkpoint['rgb_' + new_key] = v
            depth_checkpoint['depth_' + new_key] = v
        else:
            rgb_checkpoint['rgb_' + k] = v
            depth_checkpoint['depth_' + k] = v

    model.rgb_body.load_state_dict(rgb_checkpoint, strict=False)
    model.depth_body.load_state_dict(depth_checkpoint, strict=False)

    return model


def load_checkpoint(model, head, backbone_resume_path, head_resume_path, rgbd=False):

    if os.path.isfile(backbone_resume_path) and os.path.isfile(head_resume_path):
        print(colorstr('blue', f"Loading Backbone Checkpoint {backbone_resume_path}"))
        if ".onnx" in backbone_resume_path:
            raise Exception
        if rgbd:
            load_rgbd_backbone_checkpoint(model, backbone_resume_path)
        else:
            model.load_state_dict(torch.load(backbone_resume_path))

        print(colorstr('blue', f"Loading Head Checkpoint {head_resume_path}"))
        head.load_state_dict(torch.load(head_resume_path))

    elif os.path.isfile(backbone_resume_path):
        print(colorstr('blue', f"Loading ONLY Backbone Checkpoint {backbone_resume_path}"))
        if rgbd:
            if ".onnx" in backbone_resume_path:
                raise Exception
            load_rgbd_backbone_checkpoint(model, backbone_resume_path)
        else:
            state_dict = torch.load(backbone_resume_path, weights_only=True)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                state_dict = adapt_state_dict_for_backbone(state_dict)
                model.load_state_dict(state_dict)
    else:
        if len(backbone_resume_path) > 5 or len(head_resume_path) > 5:
            print(colorstr('red', f"You put in a path but there is no checkpoint found at {backbone_resume_path} or {head_resume_path}. Please Have a Check or Continue to Train from Scratch"))
            raise Exception(colorstr('red', f"You put in a path but there is no checkpoint found at {backbone_resume_path} or {head_resume_path}. Please Have a Check or Continue to Train from Scratch"))
        print(colorstr('red', f"No Checkpoint Found at {backbone_resume_path} and {head_resume_path}. Please Have a Check or Continue to Train from Scratch"))


def adapt_state_dict_for_backbone(state_dict, prefix="backbone."):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = prefix + k if not k.startswith(prefix) else k
        new_state_dict[new_key] = v
    return new_state_dict
