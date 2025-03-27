import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from merging_methods import MergingMethod


def main():
    models_to_merge = args.models_to_merge.split(",")
    print(f"Base model is: {args.base_model}")
    print(f"Models to be merged are: {models_to_merge}")
    print(f"Scaling coefficient is {args.scaling_coefficient}")
    device = "cuda" if args.use_gpu else "cpu"
    print(f"Merging conducted on {device}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    candidate_models = []
    for model_to_merge in models_to_merge:
        candidate_models.append(AutoModelForCausalLM.from_pretrained(model_to_merge, torch_dtype=torch.bfloat16).to(device))
    merging_engine = MergingMethod(merging_method_name=args.merge_method)
    if args.weight_mask_rates is not None:
        weight_mask_rates = args.weight_mask_rates.split(",")
        weight_mask_rates = [float(_) for _ in weight_mask_rates]
    else:
        weight_mask_rates = None
    exclude_param_names_regex = args.exclude_param_names_regex
    if args.exclude_param_names_regex:
        exclude_param_names_regex = args.exclude_param_names_regex.split(",")
        print(f"Following params are excluded: {exclude_param_names_regex}")
    merged_model = merging_engine.get_merged_model(
        merged_model=base_model,
        models_to_merge=candidate_models,
        exclude_param_names_regex=exclude_param_names_regex,
        param_value_mask_rate=args.param_value_mask_rate,
        scaling_coefficient=args.scaling_coefficient,
        mask_apply_method=args.mask_apply_method,
        weight_mask_rates=weight_mask_rates
    )
    print(f"Saving model to {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    merged_model = merged_model.to(torch.bfloat16)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--merge_method", type=str, required=True, default="average_merging")
    arg_parser.add_argument("--output_dir", type=str)
    arg_parser.add_argument('--base_model', type=str, help='base model')
    arg_parser.add_argument("--models_to_merge", type=str, required=True)
    arg_parser.add_argument("--exclude_param_names_regex", type=str, default=[])
    arg_parser.add_argument("--scaling_coefficient", type=float, default=1.0)
    arg_parser.add_argument("--param_value_mask_rate", type=float, default=0.8)
    arg_parser.add_argument("--use_gpu", action='store_true', default=False)
    arg_parser.add_argument("--mask_apply_method", type=str, default="average_merging")
    arg_parser.add_argument("--weight_mask_rates", type=str, default=None)
    args = arg_parser.parse_args()
    main()
