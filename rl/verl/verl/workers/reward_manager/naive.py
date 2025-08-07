# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.executor = ThreadPoolExecutor(max_workers=32)

    def compute_reward_item(self, data_with_index):
        """Compute reward for a single data item."""
        index, data_item = data_with_index
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[self.reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", None)
        extra_info_length = {"valid_response_length": valid_response_length.item()}
        if extra_info:
            extra_info.update(extra_info_length)
        else:
            extra_info = extra_info_length

        score = self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        return {
            "index": index,
            "score": score,
            "valid_response_length": valid_response_length,
            "prompt_str": prompt_str,
            "response_str": response_str,
            "ground_truth": ground_truth,
            "data_source": data_source,
        }

    def compute_group_score(self, prompt_str, results):
        score_list = []
        group_score_dict = {}
        for result in results:
            if prompt_str == result["prompt_str"]:
                score_list.append(result["score"])
        # 初始化一个字典，用于存放每个 key 的值列表
        key_values = {key: [] for key in results[0]["score"].keys()}

        # 遍历 score_list，将每个 key 的值添加到对应的列表中
        for score_item in score_list:
            for key, value in score_item.items():
                key_values[key].append(value)

        # 计算每个 key 的平均值，并存储到 group_score_dict 中
        for key, values in key_values.items():
            group_key = f"group_{key}"
            group_score_dict[group_key] = sum(values) / len(values) if values else 0.0
        return group_score_dict
        

    def __call__(self, data: DataProto, return_dict=False):
        """Process a batch of data using multiprocessing."""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # reward_extra_info = defaultdict(list)
        reward_extra_info = defaultdict(lambda: [None] * len(data))  # 按索引初始化，每个键对应一个固定长度的列表
        already_print_data_sources = {}

        data_with_index = [(index, data[index]) for index in range(len(data))]

        results = list(self.executor.map(self.compute_reward_item, data_with_index))

        prompt_str2_score = defaultdict(dict)
        # Process results
        for result in results:
            i = result["index"]
            score = result["score"]
            valid_response_length = result["valid_response_length"]
            prompt_str = result["prompt_str"]

            if isinstance(score, dict):
                reward = score["score"]
                if prompt_str not in prompt_str2_score:
                    group_score_dict = self.compute_group_score(prompt_str, results)
                    prompt_str2_score[prompt_str] = group_score_dict
                else:
                    group_score_dict = prompt_str2_score[prompt_str]

                for key, value in score.items():
                    reward_extra_info[key][i] = value  # 按索引更新 reward_extra_info
                # for key, value in group_score_dict.items():
                #     reward_extra_info[key][i] = value
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            data_source = result["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", result["prompt_str"])
                print("[response]", result["response_str"])
                print("[ground_truth]", result["ground_truth"])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
