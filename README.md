# Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging

[[PDF]](resource/MM4Long2Short.pdf)

![overall figures](resource/fig1.png)

## Summary of our findings:

- Model merging is a highly efficient approach for long-to-short reasoning, as it directly operates on model parameters **without requiring additional training**.

- Task-vector based merging methods, especially like TA and Ties-Merging, can achieve long-to-short reasoning with around **50\% length reduction** alongside **accuracy parity or even marginal gains** on 7B models. 
  
- SVD-based merging methods exhibit limited effectiveness, delivering moderate performance and serving as viable alternatives only when task vectors inherently possess low-rank spectral characteristics.
  
- Activation-based merging is the future, as it demonstrates impressive performance in terms of both reasoning accuracy (+1.9) and response length compression ratios (-49.8\%).
  
- Model merging methods applied to 1.5B-scale models remain effective on simple tasks. Smaller models struggle to learn long CoT reasoning ability through model merging. 
  
- The merging of large-scale models (14B and 32B) poses significant challenges in simultaneously maintaining reasoning performance while substantially reducing response length.


More results and code will be released soon. Stay tuned!