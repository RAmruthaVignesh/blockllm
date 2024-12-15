import torch
from transformers import Trainer
import collections
import torch.optim as optim
import logging
import numpy as np
from blockllm_torch.blockllm import AdamBCDConservative, AdamWBCDConservative
import random
from types import MethodType


logging.basicConfig(level=logging.INFO)


def get_trainable_params(model, threshold=0.001, top_k=10):
    grad_mag_param_names = [
        (param, param.grad.abs().sum().item()) for param in model.parameters()
    ]
    sorted_param_names = [
        param
        for param, _ in sorted(grad_mag_param_names, key=lambda x: x[1], reverse=True)
    ]
    top_k_params = sorted_param_names[:top_k]
    del sorted_param_names
    return top_k_params

    # for name, param in model.named_parameters():
    #     if name in top_k_param_names:
    #         yield param


def on_log(optimizer, total_no_params):
    total_updated_parameters = 0
    total_parameters = 0
    updated_parameters = optimizer.state_dict()["updated_parameters"]
    for p_name in updated_parameters:
        total_updated_parameters += updated_parameters[p_name].view(-1).sum()
        # total_parameters += updated_parameters[p_name].view(-1).shape[0]
    logging.info(
        f"Total updated parameters: {total_updated_parameters} | Total parameters: {total_no_params} | percentage_updated: {(total_updated_parameters / total_no_params) * 100}"
    )


def get_parameter_by_name(model, name):
    for param_name, param in model.named_parameters():
        if param_name == name:
            return param
    return None



def adjust_parameters_selective_weighted(
    trainer,
    model,
    batch,
    sparsity_level,
    total_no_params=None,
    param_dist_weights=None,
    learning_rate=1e-3,
    type="c4",  # or "glue"
    pad_idx=None,
    world_size=1,
    param_visit_count=None,  # Add this parameter to keep track of previously selected parameters
    optimizer=None,
    optimizer_state=None,
):
    """For the given sparsity level, identify the top-n layers
    which can be pruned. Pruning is based on magnitude. Once we decide the layer(s)
    to be pruned, redefine the optimizer with the new set of parameters.
    """
    assert total_no_params is not None, "Total number of parameters is required"
    assert param_dist_weights is not None, "Parameter distribution weights are required"
    # assert optimizer is not None, "Optimizer cannot be None"

    # Set requires_grad=True for all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Forward pass and compute loss
    # output = model(input_data)
    # loss = compute_loss(output, target)
    if type == "glue":
        output = model(**batch)
        loss = output.loss
        # loss = trainer.compute_loss(model=model, inputs=batch)

        # Backward pass
        loss.backward()
    else:
        assert pad_idx is not None, "Pad index is required for C4 dataset"
        batch = {k: v.to("cuda") for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        loss.backward()

    # Initialize sums and squared sums for gradients and weights
    sum_grad, sumsq_grad, sum_weight, sumsq_weight, sum_size, sumsq_size = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    n = 0  # Counter for the number of parameters
    total_params = 0

    # First pass: Compute sums and squared sums
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().norm().item()
            weight = param_dist_weights[name]
            size = param.numel()
            total_params += size

            sum_grad += grad_norm
            sumsq_grad += grad_norm**2
            sum_weight += weight
            sumsq_weight += weight**2
            sum_size += size
            sumsq_size += size**2
            n += 1

    # Calculate means and standard deviations
    mean_grad = sum_grad / n
    std_grad = (sumsq_grad / n - mean_grad**2) ** 0.5
    mean_weight = sum_weight / n
    std_weight = (sumsq_weight / n - mean_weight**2) ** 0.5
    mean_size = sum_size / n
    std_size = (sumsq_size / n - mean_size**2) ** 0.5

    # Apply z-score normalization with scaling factor for prioritizing gradients
    grad_priority_scale = 2.0
    size_bias_factor = 0.25
    # grad_mag_param_names_weighted = [
    #     (
    #         name,
    #         param.grad.abs().norm().item(),
    #         # (
    #         #     ((param.grad.abs().norm().item() - mean_grad) / std_grad)
    #         #     * grad_priority_scale
    #         # )
    #         # + ((param_dist_weights[name] - mean_weight) / std_weight)
    #         # + (((param.numel() - mean_size) / std_size) * size_bias_factor),
    #     )
    #     for name, param in model.named_parameters()
    #     if param.grad is not None
    # ]
    grad_mag_param_names_weighted = [
        (
            name,
            param.grad.abs().norm().item(),
        )
        for name, param in model.named_parameters()
        if param.grad is not None
    ]

    # Calculate gradients magnitude and associate with parameter names
    # The layers which has high gradient norm* distance from last layer are chosen
    # Calculate max values
    # mean_grad = torch.mean(
    #     param.grad.abs().norm().item()
    #     for _, param in model.named_parameters()
    #     if param.grad is not None
    # )
    # max_weight = max(param_dist_weights.values())
    # # Normalize and compute the weighted gradient magnitudes

    # # normalizing the distances using gradient moments
    # dist_weights = np.array(list(param_dist_weights.values()))

    # def normalize_dist_weights(grad, weightage=0.05):
    #     normalized_weights = (dist_weights - grad.mean().item()) / grad.std().item()
    #     p_names = list(param_dist_weights.keys())
    #     return dict(zip(p_names, normalized_weights * weightage))

    # grad_mag_param_names_weighted = [
    #     (
    #         name,
    #         (param.grad.abs().norm().item()) * normalize_dist_weights(param.grad)[name],
    #     )
    #     for name, param in model.named_parameters()
    #     if param.grad is not None
    # ]

    # grad_mag_param_names_weighted = [
    #     (name, param.grad.abs().norm().item() * param_dist_weights[name])
    #     for name, param in model.named_parameters()
    # ]

    # # The layers which has high gradient norm are chosen
    # grad_mag_param_names_weighted = [
    #     (name, param.grad.abs().norm().item())
    #     for name, param in model.named_parameters()
    # ]

    # Normalize the param_visit_count, which is a dictionary of counts
    # for each parameter name
    if param_visit_count is not None:
        max_count = max(param_visit_count.values())
        param_visit_count = {
            name: count / max_count for name, count in param_visit_count.items()
        }

    # Sort parameter names based on gradients magnitude
    sorted_param_names = [
        name
        for name, _ in sorted(
            grad_mag_param_names_weighted,
            key=lambda x: x[1] / param_visit_count[x[0]],
            reverse=True,
        )
    ]

    # sorted_param_names = [
    #     name
    #     for name, _ in sorted(
    #         grad_mag_param_names_weighted,
    #         key=lambda x: x[1],
    #         reverse=True,
    #     )
    # ]

    # # Shuffling the list randomly
    # random.shuffle(grad_mag_param_names_weighted)
    # sorted_param_names = [name for name, _ in grad_mag_param_names_weighted]

    # Select top k parameters
    # total_params = 0
    # for name, param in model.named_parameters():
    #     total_params += param.numel()
    num_params_to_keep = int((1 - sparsity_level) * total_params)

    top_k_params = []
    # target_modules_list = ["attn", "mlp"]

    running_param_count = 0
    for name in sorted_param_names:
        # if not any(target_key in name for target_key in target_modules_list):
        #     continue
        #     if name in prev_selected_params:
        #         continue
        param = get_parameter_by_name(model, name)
        if running_param_count < num_params_to_keep:
            top_k_params.append(name)
            running_param_count += param.numel()
        else:
            break

    # Estimate the sparsity level that the optimizer needs to achieve
    # This is the num_parameters_to_keep / running_param_count
    opt_sparsity_level = num_params_to_keep / running_param_count

    # Calculate the number of parameters that are set to requires_grad=True
    # del grad_mag_param_names_weighted
    del sorted_param_names

    # Update the trainable parameters in the optimizer by redefining the optimizer
    # past_update_parameters = (
    #     trainer.optimizer.state_dict()["updated_parameters"]
    #     if trainer.optimizer
    #     else {}
    # )
    # optimizer = AdamWBCDConservative(
    #     [get_parameter_by_name(model, x) for x in top_k_params],
    #     lr_thresh_perc=opt_sparsity_level,
    #     param_names=top_k_params,
    #     num_params_to_keep=num_params_to_keep,
    #     lr=learning_rate,
    #     updated_parameters={},
    #     weight_decay=0.0,
    #     # weight_decay=0,
    # )

    current_opt_state_dict = None
    if optimizer is not None:
        current_opt_state_dict = optimizer.state_dict()

    if optimizer is None:
        lr = learning_rate
    else:
        lr = optimizer.param_groups[0]["lr"]
    optimizer = optim.Adam(
        [get_parameter_by_name(model, x) for x in top_k_params], lr=lr
    )

    if current_opt_state_dict is not None:
        # Need to update everything except param group
        pass

        # optimizer.load_state_dict(current_opt_state_dict)
    if optimizer_state is not None:

        # # Get the current state dictionary
        # current_state = current_opt_state_dict['state']

        # # Get the new optimizer's state dictionary
        # new_state = optimizer.state_dict()

        # # Create a new dictionary to store the updated state
        # updated_state = {}

        # Go over the parameters in the optimizer
        for group in optimizer.param_groups:
            for param in group["params"]:
                # Compare their ids with current_state_dict keys
                param_id = id(param)
                if param_id in [id(p) for p in optimizer_state.keys()]:
                    # If found, copy the value from current_state_dict
                    optimizer.state[param] = optimizer_state[param]

    # for param in optimizer_state:
    #     if param in top_k_params:
    #         optimizer.state[param] = optimizer_state[param]

    #  Freeze all parameters other than the top-k parameters
    # for name, param in model.named_parameters():
    #     if name not in top_k_params:
    #         param.requires_grad = False
    # prev_selected_params.update(top_k_params)
    for name in top_k_params:
        if name in param_visit_count:
            param_visit_count[name] += 1
        else:
            param_visit_count[name] = 1

    del top_k_params
    torch.cuda.empty_cache()

    return optimizer, param_visit_count


def adjust_parameters_selective_weighted_greedy(
    trainer,
    model,
    batch,
    sparsity_level,
    total_no_params=None,
    param_dist_weights=None,
    learning_rate=1e-3,
):
    """
    Use a greedy strategy to select layers for training based on the gradient magnitude.
    Specifically, select layers that have the highest gradient magnitude until the desired
    sparsity level is reached. After that pick the layers that have the smallest number of parameters.
    """
    assert total_no_params is not None, "Total number of parameters is required"
    assert param_dist_weights is not None, "Parameter distribution weights are required"

    # Set requires_grad=True for all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Forward pass and compute loss
    # output = model(input_data)
    # loss = compute_loss(output, target)
    output = model(**batch)
    loss = output.loss

    # Backward pass
    loss.backward()

    grad_mag_param_names_weighted = [
        (
            name,
            param.grad.abs().norm().item(),
            param.numel(),
            # (
            #     ((param.grad.abs().norm().item() - mean_grad) / std_grad)
            #     * grad_priority_scale
            # )
            # + ((param_dist_weights[name] - mean_weight) / std_weight)
            # + (((param.numel() - mean_size) / std_size) * size_bias_factor),
        )
        for name, param in model.named_parameters()
        if param.grad is not None
    ]

    # Calculate gradients magnitude and associate with parameter names

    # Sort parameter names based on gradients magnitude
    sorted_param_names = [
        name
        for name, _, _ in sorted(
            grad_mag_param_names_weighted, key=lambda x: x[1], reverse=True
        )
    ]

    # Select top k parameters
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
    num_params_to_keep = int((1 - sparsity_level) * total_params)

    top_k_params = []
    running_param_count = 0

    # Greedy strategy
    for idx, name in enumerate(sorted_param_names):
        param = get_parameter_by_name(model, name)

        if running_param_count + param.numel() < num_params_to_keep:
            top_k_params.append(name)
            running_param_count += param.numel()
        else:
            # Pick the parameter in the next k layers that has the smallest number of parameters

            # Get the next k layers
            next_k_layers = sorted_param_names[idx : idx + 10]

            # Pick the layer with the smallest number of parameters
            # that just exceeds the number of parameters to keep
            next_k_params = [
                (name, param)
                for name, param in model.named_parameters()
                if (name in next_k_layers)
                and (param.numel() + running_param_count > num_params_to_keep)
            ]
            next_k_params = sorted(next_k_params, key=lambda x: x[1].numel())
            for name, param in next_k_params:
                if running_param_count < num_params_to_keep:
                    top_k_params.append(name)
                    running_param_count += param.numel()
                else:
                    break


    # At the end of the loop, the number of parameters in top_k_params is less than num_params_to_keep
    # now select the remaining parameters based on the number of parameters
    # sort parameters based on the number of parameters (numel)
    # sorted_param_names = (
    #     name
    #     for name, _, _ in sorted(
    #         grad_mag_param_names_weighted, key=lambda x: x[1]/x[2]  , reverse=True
    #     )
    # )

    # for name in sorted_param_names:
    #     if param in top_k_params:
    #         continue
    #     param = get_parameter_by_name(model, name)

    #     if running_param_count < num_params_to_keep:
    #         top_k_params.add(name)
    #         running_param_count += param.numel()
    #     else:
    #         break

    # Estimate the sparsity level that the optimizer needs to achieve
    # This is the num_parameters_to_keep / running_param_count
    opt_sparsity_level = num_params_to_keep / running_param_count

    # Calculate the number of parameters that are set to requires_grad=True
    del grad_mag_param_names_weighted
    del sorted_param_names

    optimizer = AdamBCDConservative(
        [get_parameter_by_name(model, x) for x in top_k_params],
        lr_thresh_perc=opt_sparsity_level,
        param_names=list(top_k_params),
        num_params_to_keep=num_params_to_keep,
        lr=learning_rate,
        updated_parameters={},
    )

    logging.info(f"Selected paramters for training: {top_k_params}")
    #  Freeze all parameters other than the top-k parameters
    for name, param in model.named_parameters():
        if name not in top_k_params:
            param.requires_grad = False
    del top_k_params

    torch.cuda.empty_cache()
    return optimizer
