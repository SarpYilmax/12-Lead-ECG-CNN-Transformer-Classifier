import torch
import torch.nn as nn


def load_checkpoint(model: nn.Module, checkpoint_path: str, device) -> nn.Module:
    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        model_expects_module = isinstance(model, nn.DataParallel)
        fixed_state_dict = {}
        for k, v in state_dict.items():
            if has_module_prefix and not model_expects_module:
                fixed_state_dict[k[len("module."):]] = v
            elif not has_module_prefix and model_expects_module:
                fixed_state_dict[f"module.{k}"] = v
            else:
                fixed_state_dict[k] = v
        model.load_state_dict(fixed_state_dict)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model
