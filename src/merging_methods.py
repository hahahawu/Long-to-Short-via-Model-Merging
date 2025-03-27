from collections import defaultdict, OrderedDict
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import re
import gc
from task_vector import TaskVector
from utils import get_param_names_to_merge, get_modules_to_merge
from mask_weights_utils import mask_model_weights


class MergingMethod:
    def __init__(self, merging_method_name: str):
        """
        Methods for model merging.
        :param merging_method_name: str, name of the merging method, can be "average_merging", "task_arithmetic",
         "ties_merging", "latent_merging"
        :return:
        """
        self.merging_method_name = merging_method_name

    def copy_params_to_model(self, params: dict, model: nn.Module):
        """
        copy parameters in "params" to the model
        :param params: dict, dictionary of parameters
        :param model: nn.Module, model that needs to copy parameters
        :return:
        """
        for param_name, param_value in model.named_parameters():
            if param_name in params:
                param_value.data.copy_(params[param_name])

    def average_merging(self, models_to_merge: list, exclude_param_names_regex: list):
        """
        average merging method
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :return:
        """
        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_param_dict = defaultdict(list)
        # iterate each individual model that needs to be merged
        for model_to_merge in models_to_merge:
            param_dict = {param_name: param_value for param_name, param_value in model_to_merge.named_parameters()}
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()),
                                                            exclude_param_names_regex=exclude_param_names_regex)
            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(param_dict[param_name])

        with torch.no_grad():
            # average merging of individual models' parameters
            averaged_params = {param_name: torch.stack(model_to_merge_param, dim=0).mean(dim=0) for
                               param_name, model_to_merge_param in models_to_merge_param_dict.items()}

        return averaged_params

    def task_arithmetic(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list,
                        scaling_coefficient: float = 1.0):
        """
        task arithmetic method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """
        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        merged_task_vector = None
        while len(models_to_merge):
            if merged_task_vector is None:
                merged_task_vector = TaskVector(pretrained_model=merged_model, finetuned_model=models_to_merge.pop(0),
                                                exclude_param_names_regex=exclude_param_names_regex)
            else:
                merged_task_vector += TaskVector(pretrained_model=merged_model, finetuned_model=models_to_merge.pop(0),
                                                 exclude_param_names_regex=exclude_param_names_regex)
        merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model,
                                                                         scaling_coefficient=scaling_coefficient)

        return merged_params

    def ties_merging(self, merged_model: nn.Module,
                     models_to_merge: list,
                     exclude_param_names_regex: list,
                     param_value_mask_rate: float = 0.8,
                     scaling_coefficient: float = 1.0):
        """
        Optimized ties merging method that minimizes memory usage by processing parameters individually.

        Instead of flattening all parameters across the entire model (which creates huge temporary tensors),
        this implementation iterates over each parameter tensor (key by key). For each parameter:
          1. It masks the smallest (by absolute value) elements rel. to the local tensor (using a fraction
             param_value_mask_rate). Note that this is a per-tensor approximation versus the original global
             thresholding, but saves a lot of memory.
          2. It computes an aggregated sign (using the sign of the sum of the masked tensors) and then
             preserves only the elements whose sign agrees with this aggregated sign.
          3. It averages the preserved values from the different models.
        Finally, the merged task vector is combined with the pretrained model (merged_model) using the
        provided scaling_coefficient.

        :param merged_model: nn.Module, the baseline (pre-trained) model.
        :param models_to_merge: list of models (e.g. finetuned versions) whose task vectors will be merged.
        :param exclude_param_names_regex: list of regex strings to exclude certain parameter names.
        :param param_value_mask_rate: float, fraction of values (per parameter tensor) to mask (set to 0) based on their magnitude.
        :param scaling_coefficient: float, scaling coefficient used when combining the task vector with the pretrained model.
        :return: nn.Module, merged model
        """
        # Create TaskVector objects for each model to merge.
        # (If the TaskVector implementation does deep copies, one might be able to optimize that separately.)
        task_vectors = [TaskVector(pretrained_model=merged_model,
                                   finetuned_model=model,
                                   exclude_param_names_regex=exclude_param_names_regex)
                        for model in models_to_merge]

        # Use the parameter keys from the first TaskVector; we assume all task_vectors have the same keys.
        sorted_keys = sorted(task_vectors[0].task_vector_param_dict.keys())
        merged_task_vector_param_dict = OrderedDict()

        # Process each parameter key individually.
        for key in sorted_keys:
            # Get parameter tensors from each task vector (for each model)
            param_list = [tv.task_vector_param_dict[key] for tv in task_vectors]

            masked_params = []
            # For each parameter tensor, perform local masking.
            for param in param_list:
                # Flatten the parameter tensor
                param_flat = param.view(-1)
                num_params = param_flat.numel()
                # Determine how many elements to mask in this tensor.
                k = int(num_params * param_value_mask_rate)
                if k > 0:
                    # Compute kth smallest (by absolute value) threshold.
                    # kthvalue returns a namedtuple (values, indices)
                    kth_val = param_flat.abs().kthvalue(k=k).values
                    # Create a mask: keep elements with absolute values >= kth_val.
                    mask = param.abs() >= kth_val
                    # Multiply (elementwise) to zero out the smallest ones.
                    masked_param = param * mask.to(param.dtype)
                else:
                    masked_param = param
                masked_params.append(masked_param)

            # Compute an aggregated sign per element.
            # This mirrors the behavior: aggregated_sign = sign(sum over models)
            summed = sum(masked_params)
            aggregated_sign = torch.sign(summed)
            # For any element where the sign is zero, default to positive (1.0)
            aggregated_sign[aggregated_sign == 0] = 1.0

            # For each model’s masked parameter, keep only elements whose sign
            # matches the aggregated sign.
            preserved_params = []
            for mp in masked_params:
                # Create a boolean mask for matching signs.
                sign_mask = (((aggregated_sign > 0) & (mp > 0)) |
                             ((aggregated_sign < 0) & (mp < 0))).to(mp.dtype)
                preserved = mp * sign_mask
                preserved_params.append(preserved)

            # Count how many models preserved a nonzero element for each coordinate.
            count_preserved = sum([(p != 0).float() for p in preserved_params])
            # Compute the merged parameter (average the preserved contributions)
            merged_param = sum(preserved_params) / torch.clamp(count_preserved, min=1.0)
            merged_task_vector_param_dict[key] = merged_param

        # Build the merged task vector from the merged parameter dictionary.
        merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_param_dict)
        # Combine with the base (pretrained) model using the provided scaling coefficient.
        return merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model,
                                                                scaling_coefficient=scaling_coefficient)

    def ties_merging_dare(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list,
                          param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0):
        """
        ties merging method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """

        def task_vector_param_dict_to_single_vector(task_vector: TaskVector):
            """
            convert parameter dictionary in task vector to a single vector
            :param task_vector: TaskVector, task vector
            :return:
            """
            task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
            sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))
            del task_vector_param_dict

            # Tensor, shape (num_total_params, )
            return nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])

        def single_vector_to_task_vector_param_dict(single_vector: torch.Tensor, task_vector: TaskVector):
            """
            convert a single vector to parameter dictionary in task vector
            :param single_vector: Tensor, single vector that contain all parameters in task_vector.task_vector_param_dict
            :param task_vector: TaskVector, task vector
            :return:
            """
            task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
            sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))
            del task_vector_param_dict

            nn.utils.vector_to_parameters(single_vector, sorted_task_vector_param_dict.values())

            return sorted_task_vector_param_dict

        def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor,
                                                 param_value_mask_rate: float = 0.8):
            """
            mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
            :return:
            """
            # num_models_to_merge, num_total_params = flattened_models_to_merge_param.shape
            num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)

            # Tensor, shape (num_models_to_merge, 1), find the num_mask_params-th smallest magnitude element of all the parameters in each individual model
            kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            mask = flattened_models_to_merge_param.abs() >= kth_values
            del kth_values
            flattened_models_to_merge_param = flattened_models_to_merge_param * mask
            del mask

            return flattened_models_to_merge_param

        def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
            """
            get the signs for each parameter in flattened_models_to_merge_param, computed over individual models that need to be merged
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :return:
            """
            # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
            param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
            # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
            majority_sign = torch.sign(param_signs.sum(dim=0))
            param_signs[param_signs == 0] = majority_sign
            return param_signs

        def disjoint_merge(flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor):
            """
            disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, and calculates the averaged parameters.
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :param param_signs: Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
            :return:
            """
            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | (
                    (param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
            # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
            param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask
            del param_to_preserve_mask

            # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
            num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
            # Tensor, shape (num_total_params, ), the averaged flattened parameters
            merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved,
                                                                                       min=1.0)
            del param_to_preserve

            return merged_flattened_param

        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge,
                                                   exclude_param_names_regex=exclude_param_names_regex) for
                                        model_to_merge in models_to_merge]
        del models_to_merge

        flattened_models_to_merge_param = [task_vector_param_dict_to_single_vector(task_vector=task_vector) for
                                           task_vector in models_to_merge_task_vectors]
        models_to_merge_task_vectors = models_to_merge_task_vectors[0]
        # Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        flattened_models_to_merge_param = torch.vstack(flattened_models_to_merge_param)

        with torch.no_grad():
            # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
            flattened_models_to_merge_param = mask_smallest_magnitude_param_values(
                flattened_models_to_merge_param=flattened_models_to_merge_param,
                param_value_mask_rate=param_value_mask_rate)

            # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
            param_signs = get_param_signs(flattened_models_to_merge_param=flattened_models_to_merge_param)

            # Tensor, shape (num_total_params, ), disjoint merge
            merged_flattened_param = disjoint_merge(flattened_models_to_merge_param=flattened_models_to_merge_param,
                                                    param_signs=param_signs)
            del flattened_models_to_merge_param, param_signs

            # merged parameter dictionary
            merged_task_vector_param_dict = single_vector_to_task_vector_param_dict(
                single_vector=merged_flattened_param, task_vector=models_to_merge_task_vectors)
            merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_param_dict)
            del merged_task_vector_param_dict
            # combine with parameters of the merged model based on scaling coefficient
            merged_model = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model,
                                                                            scaling_coefficient=scaling_coefficient)

        return merged_model

    def merging_models(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list,
                       scaling_coefficient: float = 1.0,
                       param_value_mask_rate: float = 0.8,
                       weight_format: str = "delta_weight", weight_mask_rates: list = None,
                       use_weight_rescale: bool = True, mask_strategy: str = "random",
                       mask_apply_method: str = "average_merging", models_use_deepcopy: bool = False):
        """
        model merging methods
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
        :param weight_mask_rates: list, list of weight mask rates
        :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
        :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
        :param mask_apply_method: str, merging method that the mask strategy applies
        :param models_use_deepcopy: boolean, whether to deepcopy the models
        :return:
        """
        if self.merging_method_name == "average_merging":
            merged_params = self.average_merging(models_to_merge=models_to_merge,
                                                 exclude_param_names_regex=exclude_param_names_regex)
        elif self.merging_method_name == "task_arithmetic":
            merged_params = self.task_arithmetic(merged_model=merged_model, models_to_merge=models_to_merge,
                                                 exclude_param_names_regex=exclude_param_names_regex,
                                                 scaling_coefficient=scaling_coefficient)
        elif self.merging_method_name == "ties_merging":
            merged_params = self.ties_merging(merged_model=merged_model, models_to_merge=models_to_merge,
                                              exclude_param_names_regex=exclude_param_names_regex,
                                              param_value_mask_rate=param_value_mask_rate,
                                              scaling_coefficient=scaling_coefficient)
        elif self.merging_method_name == "ties_merging_dare":
            merged_params = self.ties_merging_dare(merged_model=merged_model, models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=exclude_param_names_regex,
                                                   param_value_mask_rate=param_value_mask_rate,
                                                   scaling_coefficient=scaling_coefficient)
        elif self.merging_method_name == "mask_merging":
            with torch.no_grad():
                if models_use_deepcopy:
                    new_models_to_merge = copy.deepcopy(models_to_merge)
                else:
                    new_models_to_merge = models_to_merge
                for new_model_to_merge, weight_mask_rate in zip(new_models_to_merge, weight_mask_rates):
                    # for each individual model, mask its weight
                    masked_param_dict = mask_model_weights(finetuned_model=new_model_to_merge,
                                                           pretrained_model=merged_model,
                                                           exclude_param_names_regex=exclude_param_names_regex,
                                                           weight_format=weight_format,
                                                           weight_mask_rate=weight_mask_rate,
                                                           use_weight_rescale=use_weight_rescale,
                                                           mask_strategy=mask_strategy)
                    self.copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)
            if mask_apply_method == "average_merging":
                merged_params = self.average_merging(models_to_merge=new_models_to_merge,
                                                     exclude_param_names_regex=exclude_param_names_regex)
            elif mask_apply_method == "task_arithmetic":
                merged_params = self.task_arithmetic(merged_model=merged_model, models_to_merge=new_models_to_merge,
                                                     exclude_param_names_regex=exclude_param_names_regex,
                                                     scaling_coefficient=scaling_coefficient)
            elif mask_apply_method == "ties_merging":
                merged_params = self.ties_merging(merged_model=merged_model, models_to_merge=new_models_to_merge,
                                                  exclude_param_names_regex=exclude_param_names_regex,
                                                  param_value_mask_rate=param_value_mask_rate,
                                                  scaling_coefficient=scaling_coefficient)
            elif mask_apply_method == "ties_merging_dare":
                merged_params = self.ties_merging_dare(merged_model=merged_model, models_to_merge=new_models_to_merge,
                                                       exclude_param_names_regex=exclude_param_names_regex,
                                                       param_value_mask_rate=param_value_mask_rate,
                                                       scaling_coefficient=scaling_coefficient)
            else:
                raise NotImplementedError(f"unsupported for mask_apply_method {mask_apply_method}!")
        else:
            raise NotImplementedError(f"unsupported for merging_method_name {self.merging_method_name}!")
        return merged_params

    def get_merged_model(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list,
                         scaling_coefficient: float = 1.0,
                         param_value_mask_rate: float = 0.8,
                         weight_format: str = "delta_weight", weight_mask_rates: list = None,
                         use_weight_rescale: bool = True, mask_strategy: str = "random",
                         mask_apply_method: str = "average_merging", models_use_deepcopy: bool = False):
        """
        merge the parameters of models_to_merge to merged_model
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
        :param weight_mask_rates: list, list of weight mask rates
        :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
        :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
        :param mask_apply_method: str, merging method that the mask strategy applies
        :param models_use_deepcopy: boolean, whether to deepcopy the models
        :return:
        """
        # merged_params, dict of parameters
        merged_params = self.merging_models(merged_model=merged_model, models_to_merge=models_to_merge,
                                            exclude_param_names_regex=exclude_param_names_regex,
                                            scaling_coefficient=scaling_coefficient,
                                            param_value_mask_rate=param_value_mask_rate,
                                            weight_format=weight_format, weight_mask_rates=weight_mask_rates,
                                            use_weight_rescale=use_weight_rescale, mask_strategy=mask_strategy,
                                            mask_apply_method=mask_apply_method,
                                            models_use_deepcopy=models_use_deepcopy)
        self.copy_params_to_model(params=merged_params, model=merged_model)

        return merged_model
