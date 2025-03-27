import os
import sys
import json
from pathlib import Path

import numpy as np
import transformers
import re

from transformers import AutoTokenizer

transformers.utils.logging.set_verbosity_error()


def get_avg_length(model_path, result_path):
    result_path = Path(result_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    keywords = ['wait', 're-examine', 'double-check', 'let me check', 'recap', 'let me just verify', 'let me just check']
    pattern = r'|'.join(re.escape(keyword) for keyword in keywords)

    for dataset_name in os.listdir(result_path):
        print(dataset_name)
        dataset_dir = result_path / dataset_name
        for file_name in os.listdir(dataset_dir):
            if not file_name.endswith('.jsonl'):
                continue
            print(f'\t{file_name}', end='')
            length_list = []
            kw_freq_list = []
            level_length_map = {}
            level_acc_map = {}
            level_reflection_map = {}
            reflection_cnt = 0
            with open(dataset_dir / file_name) as f:
                for line_data in f:
                    line_data = json.loads(line_data)
                    length = len(tokenizer(line_data['code'][0])['input_ids'])
                    length_list.append(length)
                    keywords_match = re.findall(pattern, line_data["code"][0], re.IGNORECASE)
                    if len(keywords_match) > 0:
                        kw_freq_list.append(len(keywords_match))
                        reflection_cnt += 1
                    if 'level' in line_data:
                        level_length_map.setdefault(line_data['level'], []).append(length)
                        level_acc_map.setdefault(line_data['level'], []).append(int(line_data["score"][0]))
                        level_reflection_map.setdefault(line_data['level'], []).append(int(len(keywords_match) > 0))
            print(f'\t{round(np.mean(length_list), 2)}[{reflection_cnt}]; [{round(np.mean(kw_freq_list), 1) if kw_freq_list else 0}]')
            if level_length_map:
                for level, level_length_list in sorted(level_length_map.items()):
                    print(
                        f'\t\tlevel-{level}: {np.mean(level_length_list)};\t{round(np.mean(level_acc_map[level]) * 100, 1)};\t[{round(np.mean(level_reflection_map[level]), 3)}]')


if __name__ == '__main__':
    _, _model_path, _result_path = sys.argv
    get_avg_length(_model_path, _result_path)
