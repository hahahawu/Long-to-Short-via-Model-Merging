# Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging

[[ArXiv]](http://arxiv.org/abs/2503.20641) | [[Latest PDF]](resource/MM4Long2Short.pdf)

![overall figures](resource/fig1.png)

## News 🔔 🔔 🔔 

- 🔥 [04 June, 2025] We found the fantastic performance of TIES-Merging on shortening the output length of DeepSeek-R1-0528-Qwen3-8B.
> [!IMPORTANT]
> **Reducing around 50% length with performance improvement 4 points on AIME24! AND preserving the no_think ability!**
- 💡 [27 March, 2025] We release the code of Average Merging, Task Arithmetic, Ties-Merging and DARE. Special thanks to [MergeLM](https://github.com/yule-BUAA/MergeLM) for their great work. We use this repo as our codebase.
- 📣 [26 March, 2025] Our work is available on [[ArXiv]](http://arxiv.org/abs/2503.20641). 

## Summary of our findings 🔥🔥🔥:

- Model merging is a highly efficient approach for long-to-short reasoning, as it directly operates on model parameters **without requiring additional training**.

- Task-vector based merging methods, especially like TA and Ties-Merging, can achieve long-to-short reasoning with around **50\% length reduction** alongside **accuracy parity or even marginal gains** on 7B models. 
  
- SVD-based merging methods exhibit limited effectiveness, delivering moderate performance and serving as viable alternatives only when task vectors inherently possess low-rank spectral characteristics.
  
- Activation-based merging is the future, as it demonstrates impressive performance in terms of both reasoning accuracy (+1.9) and response length compression ratios (-49.8\%).
  
- Model merging methods applied to 1.5B-scale models remain effective on simple tasks. Smaller models struggle to learn long CoT reasoning ability through model merging. 
  
- The merging of large-scale models (14B and 32B) poses significant challenges in simultaneously maintaining reasoning performance while substantially reducing response length.

## Related Work 📑

- Average Merging: [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482)
  
- Task Arithmetic: [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)

- Ties-Merging: [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)

- DARE: [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)

- LoRE-Merging: [LoRE-Merging: Exploring Low-Rank Estimation For Large Language Model Merging](https://arxiv.org/abs/2502.10749)

- Twin-Merging: [Twin-Merging: Dynamic Integration of Modular Expertise in Model Merging](https://arxiv.org/abs/2406.15479)

- AIM: [Activation-Informed Merging of Large Language Models](https://arxiv.org/abs/2502.02421)

- Sens-Merging: [Sens-Merging: Sensitivity-Guided Parameter Balancing for Merging Large Language Models](https://arxiv.org/abs/2502.12420)

## 🔔 Early Test on DeepSeek-R1-0528-Qwen3-8B

We directly merge the R1-Qwen3-8B with the Qwen3-8B (as base) models using TIES-Merging with `k = 0.7, α = 0.7`. We sample 16 answers for each question and calculate the the average score (`generation_parameters: max_new_tokens = 32768, temperature = 0.6, top_p = 0.95, top_k = 20}`).

| Model           | R1-0528-Qwen3-8B          | Merged Qwen3-8B                 | 
|------------------|----------------|--------------------|
| AIME24  | 70.63 (11328.25)        | 74.58 (6448.5)           |

According to [Qwen3's guidelines](https://huggingface.co/Qwen/Qwen3-8B), there are two ways to achieve the switch between the /think and /no_think modes, i.e. `enable_thinking=False|True` and appending `/think | /no_think` to the instruction.

- Extensive experiments on R1-Qwen3-8B reveal that the no_think capability has been completely diminished in both modes, resulting in excessively lengthy responses.
- Our merged model perserves the no_think ability according to our testing. We found that the model can directly generate the answer part by setting `enable_thinking=False`. However, the alternative switch mode triggered by the `/no_think` keyword appears to fail in most cases.

*Stay tuned to our more new results!*

## Environments
For merging methods:
```angular2html
numpy==1.26.4
torch==2.5.1
transformers==4.48.2
```

For the evaluation, we recommend following the usage of [[Qwen2.5-Math Eval Toolkit]](https://github.com/QwenLM/Qwen2.5-Math).

## Implementation

Our implementation is adapted from [MergeLM](https://github.com/yule-BUAA/MergeLM). We optimize the code of some merging methods, such as TA and Ties-Merging, for computational efficiency.

### Average merging

```shell
python src/main_merging.py --merge_method average_merging \
                           --output_dir DIR \
                           --base_model MODEL_PATH \
                           --models_to_merge MODEL_PATH1,MODEL_PATH2,...,MODEL_PATHn
```

### Task Arithmetic
```shell
python src/main_merging.py --merge_method task_arithmetic \
                           --output_dir DIR \
                           --base_model MODEL_PATH \
                           --models_to_merge MODEL_PATH1,MODEL_PATH2,...,MODEL_PATHn \
                           --scaling_coefficient α
```

### Ties-Merging
```shell
python src/main_merging.py --merge_method ties_merging/ties_merging_dare \
                           --output_dir DIR \
                           --base_model MODEL_PATH \
                           --models_to_merge MODEL_PATH1,MODEL_PATH2,...,MODEL_PATHn \
                           --scaling_coefficient α \
                           --param_value_mask_rate k
```

Note: You can choose to use `ties_merging` that we optimize the implementation for better computational efficiency or `ties_merging_dare` which is the original implementation in MergeLM. We have compared two implementations and find the results are comparable.

### DARE
```shell
python src/main_merging.py --merge_method mask_merging \
                           --output_dir DIR \
                           --base_model MODEL_PATH \
                           --models_to_merge MODEL_PATH1,MODEL_PATH2,...,MODEL_PATHn \
                           --scaling_coefficient α \
                           --param_value_mask_rate k \
                           --mask_apply_method [average_merging || task_arithmetic || ties_merging || ties_merging_dare] \
                           --weight_mask_rates p
```

### AIM & Sens-Merging
We will release the code of activation-based methods soon. Stay tuned!

### Evaluations

```shell
git clone https://github.com/QwenLM/Qwen2.5-Math.git
cd Qwen2.5-Math-main/evaluation
```

Move `src/evaluation/data_process.py` and `src/evaluation/l2s_eval.sh` as following:

```markdown
Qwen2.5-Math-Main/evaluation
├── sh
│   ├── l2s_eval.sh
├── data
│   ├── math500
├── outputs
│   ├── ...
├── data_process.py
└── ...
```

Note: [MATH500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) are not in original database. You should manually add it to `Qwen2.5-Math-Main/evaluation/data`.

Run the evaluation:
```shell
CUDA_VISIBLE_DEVICES="0,1,2,3" bash sh/l2s_eval.sh [PROMPT_TYPE:qwen25-math-cot] [MODEL_PATH] [MAX_TOKEN:10240] [NUM_SHOTS:0] [DATASETS:aime24,math500,gsm8k,college_math,minerva_math,olympiadbench]
```

To make sure the reproducibility of the results, we set `temperature=0`, `top_p=1`.

## Configurations

| Method           | 1.5B           | 7B                 | 14B                | 32B                |
|------------------|----------------|--------------------|--------------------|--------------------|
| Task Arithmetic  | α = 0.7        | α = 0.7           | α = 0.7           | α = 0.7           |
| Ties-Merging     | k = 0.8, α = 1.0 | k = 0.8, α = 1.0   | k = 0.2, α = 0.5   | k = 0.25, α = 0.55 |
| DARE             | p = 0.3        | p = 0.3           | p = 0.4           | -                  |
| AIM-Ties         | ω = 0.4        | ω = 0.4           | ω = 0.4           | -                  |
| Sens-Merging     | α = 0.4, T = 3.0 | α = 0.7, T = 2.0  | α = 0.8, T = 6.0  | -                  |

*Table: The hyper-parameters of various merging methods. α means the coefficient in TA merging. p means the drop rate in DARE. k denotes the trim ratio in Ties-Merging. ω means the balance factor in AIM. T is the temperature in Sens-Merging.*


## Citation
```
@article{wu2025unlockingefficientlongtoshortllm,
      title={Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging}, 
      author={Han Wu and Yuxuan Yao and Shuqi Liu and Zehua Liu and Xiaojin Fu and Xiongwei Han and Xing Li and Hui-Ling Zhen and Tao Zhong and Mingxuan Yuan},
      year={2025},
      eprint={2503.20641},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.20641}, 
}
```
