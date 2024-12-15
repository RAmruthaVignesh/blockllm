from types import MethodType
from transformers import TrainerCallback
from transformers.utils import logging
from typing import Dict, Any, Optional
import torch

logger = logging.get_logger(__name__)

class BlockLLMCallback(TrainerCallback):
    """ Sets up gradient checkpointing for BlockLLM. Modified from BAdam's implementation. Thanks to BAdam! """
    def __init__(self, *args, **kwargs):
        self.init_loss_scale = kwargs.get("init_loss_scale", 12)
        
        
    def on_train_begin(self, *args, **kwargs):
        model = kwargs["model"]

        if hasattr(model, "disable_input_require_grads") and hasattr(model, "_require_grads_hook"):
            model.disable_input_require_grads()
            logger.info("Disable embedding output's require_grads for block-wise optimizer. Instead, "
                        "set input of checkpoint layer's `requires_grad` to True when the checkpoint layer is trainable")

        model.gradient_checkpointing_enable = MethodType(gradient_checkpointing_enable_for_bcd, model)

    
def gradient_checkpointing_enable_for_bcd(
    self: "PreTrainedModel", gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None
) -> None:
    r"""
    Modification of the original method to enable gradient checkpointing for block-wise optimizer.

    To backward Pytorch checkpoint layer, the input of the backward layer should be a tensor 
    with `requires_grad=True`. In full parameter training scheme, the output of the embedding's 
    `requires_grad` is set to True (by model.enable_input_require_grads()). However, when using 
    Block-wise update, backward to the embedding layer is not necessary and induces additional
    memory and time cost. Therefore, we disable the `requires_grad` of the embedding layer's output,
    and apply this function to the make input's `requires_grad`  to True when the checkpoint layer 
    has trainable parameters.
    """
    from torch.utils.checkpoint import checkpoint
    from functools import partial
    import inspect

    if not self.supports_gradient_checkpointing:
        raise ValueError("{} does not support gradient checkpointing.".format(self.__class__.__name__))

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}

    gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

    def custom_gradient_checkpointing_func(func, *args, **kwargs):
        module: "torch.nn.Module" = func.__self__

        if any(param.requires_grad for param in module.parameters()):
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    arg.requires_grad_(True)

        return gradient_checkpointing_func(func, *args, **kwargs)

    if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:  # old GC format
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        self.enable_input_require_grads()
        logger.warning("You are using the old GC format, some features (e.g. BAdam) will be invalid.")
    else:  # have already enabled input require gradients
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=custom_gradient_checkpointing_func)


