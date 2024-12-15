import math
import torch
from torch.optim.optimizer import Optimizer


class AdamBCDConservative(Optimizer):
    """
    NOTE: The mask variable is updated externall by the training code.
    The mask is reset at a pre-defined
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        hypergrad_lr=1e-8,
        lr_thresh_perc=0.5,
        num_params_to_keep=None,
        param_names=None,
        updated_parameters={},
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hypergrad_lr=hypergrad_lr,
            lr_thresh_perc=lr_thresh_perc,
            num_params_to_keep=num_params_to_keep,
            mask={},
            param_names=param_names,
        )
        super(AdamBCDConservative, self).__init__(params, defaults)
        self.updated_parameters = (
            updated_parameters  # Keeps the list of parameters that were updated
        )

    def load_state_dict(self, state_dict):
        # Call the base method first to load the optimizer state
        super().load_state_dict(state_dict)

        # Load the extra information
        self.updated_parameters = state_dict.get("updated_parameters", {})

    def state_dict(self):
        # Get the base state dictionary
        state_dict = super().state_dict()

        # Add the extra information to the state dictionary
        state_dict["updated_parameters"] = self.updated_parameters

        return state_dict

    # def _compute_mask(self, pname, grad, quantile):
    #     """Compute the mask based on the threshold criterion."""
    #     if pname in self.defaults["mask"]:
    #         return self.defaults["mask"][pname]
    #     else:
    #         threshold = torch.quantile(torch.abs(grad), quantile)
    #         mask = torch.gt(torch.abs(grad), threshold).float()
    #         mask = mask.to_sparse()
    #         self.defaults["mask"][pname] = mask
    #         return mask
    #         # threshold = torch.quantile(torch.abs(grad), quantile)
    #         # mask = torch.gt(torch.abs(grad), threshold).float()
    #         # self.defaults['mask'] = mask
    #         # return mask

    # def _compute_mask(self, pname, grad, quantile, sample_size=1000000):
    #     """Compute the mask based on the threshold criterion using approximate quantile computation."""
    #     if pname in self.defaults["mask"]:
    #         return self.defaults["mask"][pname]
    #     else:
    #         # Flatten the gradient tensor and randomly sample a subset of elements
    #         flat_grad = torch.abs(grad).flatten()
    #         sample_indices = torch.randperm(flat_grad.size(0))[:sample_size]
    #         sampled_grad = flat_grad[sample_indices]

    #         # Compute the quantile on the sampled subset
    #         threshold = torch.quantile(sampled_grad, quantile)

    #         # Apply the threshold to create the mask
    #         mask = torch.gt(torch.abs(grad), threshold).float()
    #         mask = mask.to_sparse()
    #         self.defaults["mask"][pname] = mask
    #         return mask

    def _compute_mask(self, pname, grad, quantile, k):
        """Compute the mask by selecting the top k gradients."""
        if pname in self.defaults["mask"]:
            return self.defaults["mask"][pname]
        else:
            # Flatten the gradient tensor and get the absolute values
            k = min(k, len(grad.view(-1)))
            flat_grad = torch.abs(grad).flatten()
            _, top_k_indices = torch.topk(flat_grad, k)
            # Initialize a mask of zeros
            mask = torch.zeros_like(flat_grad, dtype=torch.bool)

            # Set the top k positions to 1
            mask[top_k_indices] = 1
            # Reshape the mask to the original gradient shape and convert to sparse
            mask = mask.view_as(grad)
            self.defaults["mask"][pname] = mask
            return mask

    def _bookkeep_mask(self, curr_mask: torch.Tensor, p_name):
        # if p_name not in self.updated_parameters:
        if self.updated_parameters.get(p_name) is None:
            self.updated_parameters[p_name] = curr_mask

        else:
            self.updated_parameters[p_name] = (
                curr_mask.bool() | self.updated_parameters[p_name]
            )

    def step(self, closure=None, update_params=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            update_params (list, optional): List of parameters to update. If None, updates all parameters.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Use specified parameters or default to all parameters if none specified
        # params_to_update = update_params if update_params is not None else self.param_groups[0]['params']

        # for p in update_params:
        k_remaining = self.defaults["num_params_to_keep"]
        for group in self.param_groups:
            for idx, param in enumerate(group["params"]):
                pname = self.defaults["param_names"][idx]
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients.")

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param.data)
                    state["exp_avg_sq"] = torch.zeros_like(param.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = self.defaults["betas"]

                state["step"] += 1

                if self.defaults["weight_decay"] != 0:
                    grad.add_(self.defaults["weight_decay"], param.data)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(self.defaults["eps"])

                step_size = (
                    self.defaults["lr"]
                    * math.sqrt(1 - beta2 ** state["step"])
                    / (1 - beta1 ** state["step"])
                )

                # Mask computation and application
                mask = self._compute_mask(
                    pname,
                    exp_avg / denom,
                    self.defaults["lr_thresh_perc"],
                    k_remaining,
                )
                k_remaining -= len((exp_avg / denom).view(-1))
                # self._bookkeep_mask(mask, pname)  # Update the list of parameters

                param.data.addcdiv_(-step_size, exp_avg.mul(mask), denom)

        return loss
