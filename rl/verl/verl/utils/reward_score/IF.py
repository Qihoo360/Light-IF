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

import re
import json
import math
import os
import re
import sys
import requests
import scipy.stats
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import random
import json

def format_res(text):
    try:
        d = json.loads(text)
        d['context'] = {}
        d['context']['code'] = 0
        d['data'] = {}
        if 'choices' in d and len(d['choices']) > 0 and 'message' in d['choices'][0] and 'content' in d['choices'][0]['message']:
            d['data']['text'] = d['choices'][0]['message']['content']
            d['data']['reasoning_content'] = d['choices'][0]['message'].get('reasoning_content', "")
        else:
            d['data']['text'] = None
            d['context']['code'] = d.get('errno', -1)
            if 'error' in d and 'code' in d['error']:
                d['context']['code'] = d['error']['code']

            return [d['data']['text'].strip(), ""]
        return [d['data']['text'].strip(), d['data']['reasoning_content'].strip()]
    except:
        return None

def prompt_jqzl_en(question, answer):
    base_prompt = "Please act as a reviewer and evaluate the quality of the model's responses as an AI assistant. \n"
    base_prompt += "Your evaluation should prioritize the **fluency and readability** of the final answer. If the response is crafted solely to meet specific constraints and sacrifices natural flow or clarity, it should be rated poorly. For example, if the instruction requires the letter \"n\" to appear at least three times, and the response artificially includes multiple standalone instances of the letter \"n\" just to fulfill this requirement, such a response should receive a low score. Next, consider whether the response **fulfills the instruction requirements**, and evaluate it strictly according to the **scoring criteria below**: \n"
    base_prompt += "1) Score: 1–2, **Criteria:** The response is of poor quality, lacks fluency, and is written solely to satisfy the instruction constraints without regard for the overall quality of the reply. \n"
    base_prompt += "2) Score: 3–5, **Criteria:** The response is generally readable but lacks overall fluency and smoothness. \n"
    base_prompt += "3) Score: 6–8, **Criteria:** The response is of relatively high quality, with generally fluent and coherent language. \n"
    base_prompt += "4) Score: 9–10, **Criteria:** The response is of high quality, with fluent, natural, and well-structured language. \n"
    base_prompt += "Please provide the following: 1). A brief explanation to evaluate the quality of the AI assistant's response. If there are any issues with response quality or language fluency, please identify and briefly explain them；\n"
    base_prompt += "2) Then provide an evaluation score, which must be strictly graded according to the following format：\"[[rating]]\"，example：\"Score:[[4]]\".\n"
    base_prompt += "[Question]\n"
    base_prompt += "{0}\n"
    base_prompt += "[The Start of Assistant's Answer]\n"
    base_prompt += "{1}\n"
    base_prompt += "[The End of Assistant's Answer]"
    prompt = base_prompt.format(question, answer)

    return prompt

def validate_article(article, require_dic):
    # 关键词, 如果low_num<=0，max_num<=0，表示不能出现这个词语
    # 如果low_num<0，表示没有最少多少字要求
    # 如果max_num<0，表示没有最多多少字要求
    # 如果low_num > max_num，表示没有最多多少字要求
    # 如果low_num==max_num，表示只能出现max_num次
    def split_chinese_sentences(text):
        """
        将中文文本按句号、问号、感叹号分割为句子列表

        参数:
            text (str): 输入的中文文本

        返回:
            list: 分割后的句子列表（过滤空字符串）
        """
        # 正则表达式模式：匹配中文句号、问号、感叹号中的任意一个
        pattern = r'[。？！]'

        # 使用正则表达式分割文本
        raw_sentences = re.split(pattern, text)

        # 过滤分割结果中的空字符串（处理文本开头/结尾有标点或连续标点的情况）
        filtered_sentences = [sentence.strip() for sentence in raw_sentences if sentence.strip()]

        return filtered_sentences

    keywords = require_dic.get("keywords", [])
    keywords_num = require_dic.get("keywords_num", [])
    keywords_req = require_dic.get("keywords_req", [])
    if keywords_req and keywords and keywords_num and len(keywords) == len(keywords_num):
        for k, v in zip(keywords_num, keywords):
            low_num, max_num = k[0], k[1]
            cnt = article.count(v)
            if low_num > 0 and low_num <= max_num:
                if cnt < low_num:
                    return False, "keywords {} < {}".format(v, low_num)
            if max_num > 0 and max_num > low_num:
                if cnt > max_num:
                    return False, "keywords {} > {}".format(v, max_num)
            if low_num > 0 and low_num == max_num:
                if cnt != low_num:
                    return False, "keywords {} != {}".format(v, max_num)
            if low_num <= 0 and max_num <= 0:
                if cnt > 0:
                    return False, "keywords {} != {}".format(v, max_num)

    # 段落要求几个"\n", -1表示没有要求
    paragraphs_split = require_dic.get("paragraphs_split", -1)
    paragraphs_num = require_dic.get("paragraphs_num", -1)
    topic_style = require_dic.get("topic_style", 0)
    if paragraphs_split > 0 and paragraphs_num > 0:
        if paragraphs_split == 2:
            split = '\n\n'
        else:
            split = '\n\n\n'

        paragraphs = len(article.split(split))
        if topic_style in [1, 2, 3]: # 有文章标题，段落数不应该把标题包含在内
            paragraphs = paragraphs - 1
        if paragraphs != paragraphs_num:
            return False, "paragraphs {} != {}".format(paragraphs, paragraphs_num)

    # 句子个数要求，-1表示没有要求
    sentence_num = require_dic.get("sentence_num", -1)
    sentences = split_chinese_sentences(article)
    if len(sentences) != sentence_num:
        return False, "sentence_num {} != {}".format(len(sentences), sentence_num)

    # 字数要求
    # low_word_cnt<0，表示没有最少字数要求
    # max_word_cnt < 0，表示没有最多字数要求
    # low_word_cnt<0, max_word_cnt<0, 表示没有字数要求
    low_word_cnt = require_dic.get("low_word_cnt", -1)
    max_word_cnt = require_dic.get("max_word_cnt", -1)
    n = len(article)
    if low_word_cnt > 0 and n < low_word_cnt:
        return False, "low_word_cnt {} < {}".format(n, low_word_cnt)
    if max_word_cnt > 0 and n > max_word_cnt:
        return False, "max_word_cnt {} > {}".format(n, max_word_cnt)

    # 文章标题形式，0表示没有要求
    # 为1: 《》
    # 为2：[]
    # 为3：【】
    topic_style = require_dic.get("topic_style", 0)
    style = []
    if topic_style == 1:
        style = ["《", "》"]
    elif topic_style == 2:
        style = ["[", "]"]
    elif topic_style == 3:
        style = ["【", "】"]
    if style and ((style[0] not in article) or (style[1] not in article)):
        return False, "topic_style not != {}".format(style[0])

    # 开始结尾, 为空表示没有要求
    begin_content = require_dic.get("begin_content", "")
    end_content = require_dic.get("end_content", "")

    index = 0
    if style:
        index = article.index(style[1])

    b1 = article[index+1:].strip().strip("。")
    b2 = article.strip().strip("。")
    begin_content = begin_content.strip().strip("。")
    end_content = end_content.strip().strip("。")
    if begin_content != "" and not (b1.startswith(begin_content) or b2.startswith(begin_content)):
        return False, "begin_content != {}".format(begin_content)
    if end_content != "" and not article.strip().strip("。").endswith(end_content):
        return False, "end_content != {}".format(end_content)

    # 文章标题字数
    # max_topic_word_cnt==0，表示没有最多字数要求
    if style:
        max_topic_word_cnt = require_dic.get("max_topic_word_cnt", 0)
        if max_topic_word_cnt > 0:
            pattern = style[0] + "(.*?)" + style[1]
            match = re.search(pattern, article)
            if match and len(match.group(1)) > max_topic_word_cnt:
                return False, "max_topic_word_cnt {} > {}".format(len(match.group(1)), max_topic_word_cnt)

    return True, "sucess"

def validate_article_step_score(article, require_dic):
    # 关键词, 如果low_num<=0，max_num<=0，表示不能出现这个词语
    # 如果low_num<0，表示没有最少多少字要求
    # 如果max_num<0，表示没有最多多少字要求
    # 如果low_num > max_num，表示没有最多多少字要求
    # 如果low_num==max_num，表示只能出现max_num次
    def split_chinese_sentences(text):
        """
        将中文文本按句号、问号、感叹号分割为句子列表

        参数:
            text (str): 输入的中文文本

        返回:
            list: 分割后的句子列表（过滤空字符串）
        """
        # 正则表达式模式：匹配中文句号、问号、感叹号中的任意一个
        pattern = r'[。？！]'

        # 使用正则表达式分割文本
        raw_sentences = re.split(pattern, text)

        # 过滤分割结果中的空字符串（处理文本开头/结尾有标点或连续标点的情况）
        filtered_sentences = [sentence.strip() for sentence in raw_sentences if sentence.strip()]

        return filtered_sentences

    reward_score_all = 0.0

    keywords = require_dic.get("keywords", [])
    keywords_num = require_dic.get("keywords_num", [])
    keywords_req = require_dic.get("keywords_req", [])
    if keywords_req and keywords and keywords_num and len(keywords) == len(keywords_num):
        for k, v in zip(keywords_num, keywords):
            low_num, max_num = k[0], k[1]
            cnt = article.count(v)
            if low_num > 0 and max_num > low_num:
                if cnt >= low_num and cnt <= max_num:  # [low_num, max_num]
                    reward_score_all += 0.1
            elif low_num > 0 and max_num < low_num: # >= low_num
                if cnt >= low_num:
                    reward_score_all += 0.05
            elif low_num > 0 and low_num == max_num:  # == low_num
                if cnt == low_num:
                    reward_score_all += 0.1
            elif low_num <= 0 and max_num > 0:  # <= max_num
                if cnt <= max_num:
                    reward_score_all += 0.05
            elif low_num <= 0 and max_num <= 0:
                if cnt == 0:
                    reward_score_all += 0.05

    # 段落要求几个"\n", -1表示没有要求
    paragraphs_split = require_dic.get("paragraphs_split", -1)
    paragraphs_num = require_dic.get("paragraphs_num", -1)
    topic_style = require_dic.get("topic_style", 0)
    if paragraphs_split > 0 and paragraphs_num > 0:
        if paragraphs_split == 2:
            split = '\n\n'
        else:
            split = '\n\n\n'

        paragraphs = len(article.split(split))
        if topic_style in [1, 2, 3]: # 有文章标题，段落数不应该把标题包含在内
            paragraphs = paragraphs - 1
        if paragraphs != paragraphs_num:
            reward_score_all += 0.0
        else:
            reward_score_all += 0.1

    # 句子个数要求，-1表示没有要求
    sentence_num = require_dic.get("sentence_num", -1)
    sentences = split_chinese_sentences(article)
    if len(sentences) != sentence_num:
        reward_score_all += 0.0
    else:
        reward_score_all += 0.2

    # 字数要求
    # low_word_cnt<0，表示没有最少字数要求
    # max_word_cnt < 0，表示没有最多字数要求
    # low_word_cnt<0, max_word_cnt<0, 表示没有字数要求
    low_word_cnt = require_dic.get("low_word_cnt", -1)
    max_word_cnt = require_dic.get("max_word_cnt", -1)
    n = len(article)
    if low_word_cnt > 0 and max_word_cnt > low_word_cnt:
        if n >= low_word_cnt and n <= max_word_cnt:
            reward_score_all += 0.1
    elif low_word_cnt > 0:
        if n >= low_word_cnt:
            reward_score_all += 0.05
    elif low_word_cnt <= 0 and max_word_cnt > 0:
        if n <= max_word_cnt:
            reward_score_all += 0.05

    # 文章标题形式，0表示没有要求
    # 为1: 《》
    # 为2：[]
    # 为3：【】
    topic_style = require_dic.get("topic_style", 0)
    style = []
    if topic_style == 1:
        style = ["《", "》"]
    elif topic_style == 2:
        style = ["[", "]"]
    elif topic_style == 3:
        style = ["【", "】"]
    if style and ((style[0] not in article) or (style[1] not in article)):
        reward_score_all += 0.0
    else:
        reward_score_all += 0.02

    # 开始结尾, 为空表示没有要求
    begin_content = require_dic.get("begin_content", "")
    end_content = require_dic.get("end_content", "")

    index = 0
    if style:
        index = article.index(style[1])

    b1 = article[index+1:].strip().strip("。")
    b2 = article.strip().strip("。")
    begin_content = begin_content.strip().strip("。")
    end_content = end_content.strip().strip("。")
    if begin_content != "" and not (b1.startswith(begin_content) or b2.startswith(begin_content)):
        reward_score_all += 0.0
    else:
        reward_score_all += 0.02
    if end_content != "" and not article.strip().strip("。").endswith(end_content):
        reward_score_all += 0.0
    else:
        reward_score_all += 0.02

    # 文章标题字数
    # max_topic_word_cnt==0，表示没有最多字数要求
    if style:
        max_topic_word_cnt = require_dic.get("max_topic_word_cnt", 0)
        if max_topic_word_cnt > 0:
            pattern = style[0] + "(.*?)" + style[1]
            match = re.search(pattern, article)
            if match and len(match.group(1)) > max_topic_word_cnt:
                reward_score_all += 0.0
            else:
                reward_score_all += 0.05

    return reward_score_all

def judge_answer(inputs):
    answer = inputs["messages"][1]["content"].replace("\n", "\\n").replace('"', '\"')
    s = json.loads(inputs["check_code"].strip("```json\n").strip("```"))
    check_code = s["check_code"]
    if check_code.count("{") != check_code.count("}"):
        check_code = check_code.strip("}")

    answer = answer.replace("\'", "\\'")

    try:
        code_str = check_code + '\n\nresponse = ' + "'" + answer + "'" + "\nresult = check_response(response)\n"

        # 执行代码字符串
        exec(code_str)
    except:
        code_str = check_code + '\n\nresponse = ' + "\"" + answer + "\"" + "\nresult = check_response(response)\n"

        # 执行代码字符串
        exec(code_str)

    # 获取结果
    res = locals().get('result')
    return res

def validate_article_en(article, prompt, require_dic):

    # 重复指令
    prompt_to_repeat = require_dic.get("prompt_to_repeat", -1)
    if prompt_to_repeat != -1:
        s = "First repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)."
        prompt = prompt.replace(s, "").strip()
        if not article.startswith(prompt):
            return False, "prompt_to_repeat err!"
        article = article.replace(prompt, "").strip()

    # 高亮
    num_highlights = require_dic.get("num_highlights", -1)
    if num_highlights > 0:
        highlights = re.findall(r'(\*[^*]+\*|_[^_]+_)', article)
        if len(highlights) < num_highlights:
            return False, "num_highlights {} < {}".format(len(highlights), num_highlights)

    # 文章单词数量
    low_word_cnt = require_dic.get("low_word_cnt", -1)
    max_word_cnt = require_dic.get("max_word_cnt", -1)
    words = re.findall(r'\b\w+\b', article)
    word_count = len(words)
    if low_word_cnt > 0 and max_word_cnt > low_word_cnt:
        if word_count < low_word_cnt or word_count > max_word_cnt:
            return False, "num_words {} not in {}, {}".format(word_count, low_word_cnt, max_word_cnt)
    elif low_word_cnt > 0:
        if word_count < low_word_cnt:
            return False, "num_words {} < {}".format(word_count, low_word_cnt)
    elif low_word_cnt <= 0 and max_word_cnt > 0:
        if word_count > max_word_cnt:
            return False, "num_words {} > {}".format(word_count, max_word_cnt)

    # 占位符数量
    num_placeholders = require_dic.get("num_placeholders", -1)
    if num_placeholders > 0:
        placeholders = re.findall(r'\[[^\[\]]+\]', article)
        placeholder_count = len(placeholders)
        if placeholder_count < num_placeholders:
            return False, "num_placeholders {} < {}".format(placeholder_count, num_placeholders)

    # 要点个数(这个和num_highlights只能2选1）,<=0表示没有要求
    num_bullets = require_dic.get("num_bullets", -1)
    if num_bullets > 0:
        bullet_points = [line for line in article.split('\n') if line.startswith('* ')]
        if len(bullet_points) != num_bullets:
            return False, "num_bullets {} != {}".format(len(bullet_points), num_bullets)

    # 全大写单词数量, 如果max_capital_cnt<=0，low_capital_cnt<=0，表示没有要求
    # low_capital_cnt<=0，表示没有最少
    # max_capital_cnt<=0，表示没有最多
    # low_capital_cnt > max_capital_cnt，表示没有最多
    # low_capital_cnt==max_capital_cnt，表示只能出现max_num次
    low_capital_cnt = require_dic.get("low_capital_cnt", -1)
    max_capital_cnt = require_dic.get("max_capital_cnt", -1)
    words = article.split()
    caps_count = sum(1 for word in words if word.isupper() and len(word) >= 2)
    if low_capital_cnt > 0 and max_capital_cnt > low_capital_cnt:
        if caps_count > max_capital_cnt or caps_count < low_capital_cnt:
            return False, "capital_cnt {} not in {}, {}".format(caps_count, low_capital_cnt, max_capital_cnt)
    elif low_capital_cnt > 0 and max_capital_cnt < low_capital_cnt:
        if caps_count < low_capital_cnt:
            return False, "capital_cnt {} < {}".format(caps_count, low_capital_cnt)
    elif low_capital_cnt > 0 and low_capital_cnt == max_capital_cnt:
        if caps_count != low_capital_cnt:
            return False, "capital_cnt {} != {}".format(caps_count, low_capital_cnt)
    elif low_capital_cnt <= 0 and max_capital_cnt > 0: 
        if caps_count > max_capital_cnt:
            return False, "capital_cnt {} > {}".format(caps_count, max_capital_cnt)

    # 关键词, 如果所有的low_num<0，max_num<0，表示没有关键词要求
    # 如果low_num<0，表示没有最少多少字要求
    # 如果max_num<0，表示没有最多多少字要求
    # 如果low_num > max_num，表示没有最多多少字要求
    # 如果low_num==max_num，表示只能出现max_num次
    keywords = require_dic.get("keywords", [])
    keywords_num = require_dic.get("keywords_num", [])
    find_no_apper = []
    if keywords and keywords_num and len(keywords) == len(keywords_num):
        res_ = []
        for k, v in zip(keywords_num, keywords):
            low_num, max_num = k[0], k[1]
            cnt = article.count(v)
            if low_num > 0 and max_num > low_num:
                if cnt < low_num or cnt > max_num:
                    return False, "keywords_num {} not in {}, {}".format(cnt, low_num, max_num)
            elif low_num > 0 and max_num < low_num:
                if cnt < low_num:
                    return False, "keywords_num {} < {}".format(cnt, low_num)
            elif low_num > 0 and low_num == max_num:
                if cnt != low_num:
                    return False, "keywords_num {} != {}".format(cnt, low_num)
            elif low_num <= 0 and max_num > 0:
                if cnt > max_num:
                    return False, "keywords_num {} > {}".format(cnt, max_num)
            elif low_num <= 0 and max_num <= 0:
                if cnt > 0:
                    return False, "keywords_num {} > {}".format(cnt, 0)

    # 句子个数，-1表示没有要求
    num_sentences = require_dic.get("num_sentences", -1)
    if num_sentences > 0:
        sentences_style = require_dic.get("sentences_style", -1)
        sentence_endings = re.compile(r'[.!?]')
        # 移除换行和多余空格后进行计数
        sentences = sentence_endings.split(article.strip())
        # 去除空白句子
        cnt = len([s for s in sentences if s.strip()])
        if sentences_style == -1:
            if cnt >= num_sentences:
                return False, "num_sentences {} >= {}".format(cnt, num_sentences)
        elif sentences_style == 0:
            if cnt != num_sentences:
                return False, "num_sentences {} != {}".format(cnt, num_sentences)
        elif sentences_style == 1:
            if cnt <= num_sentences:
                return False, "num_sentences {} <= {}".format(cnt, num_sentences)

    # 字母要求, low_letter_num>0: 至少出现多少次；max_letter_num>0: 最多出现多少次
    # max_letter_num<=0,不能出现
    low_letter_num = require_dic.get("low_letter_num", -1)
    max_letter_num = require_dic.get("max_letter_num", -100)
    letter = require_dic.get("letter", "")
    cnt = article.count(letter)
    if low_letter_num > 0 and letter != "":
        if cnt < low_letter_num:
            return False, "letter_num {} < {}".format(cnt, low_letter_num)
    elif max_letter_num > 0 and letter != "":
        if cnt > max_letter_num:
            return False, "letter_num {} > {}".format(cnt, max_letter_num)
    elif max_letter_num < 0 and max_letter_num != -100 and letter != "":
        if cnt > 0:
            return False, "letter_num {} > {}".format(cnt, 0)

    # 段落要求
    # 几个段落，-1表示没有要求
    paragraphs = require_dic.get("paragraphs", "")
    paragraphs_num = require_dic.get("paragraphs_num", -1)
    if paragraphs_num > 0:
        if paragraphs == "\n\n" or paragraphs == "\n\n\n" or paragraphs == "***":
            paragraphs_list = [k for k in article.split(paragraphs) if k.strip() != ""]
            cnt = len(paragraphs_list)
            if cnt != paragraphs_num:
                return False, "paragraphs_num {} != {}".format(cnt, paragraphs_num)

    # 结尾, 为空表示没有要求
    end_phrase = require_dic.get("end_phrase", "")
    if end_phrase != "":
        if not article.endswith(end_phrase):
            return False, "not ends with {}".format(end_phrase)

    # 特定段落要求, 如果nth_paragraph>0, first_word作用在段落上
    nth_paragraph = require_dic.get("nth_paragraph", -1)
    if nth_paragraph > 0:
        first_word = require_dic.get("first_word", "")
        if first_word != "":
            paragraphs_list = [k for k in article.split(paragraphs) if k.strip() != ""]
            if len(paragraphs_list) < nth_paragraph or (not paragraphs_list[nth_paragraph-1].startswith(first_word)):
                return False, "paragraphs {} not startswith {}".format(nth_paragraph, first_word)
            
    return True, "sucess"

def validate_article_step_score_en(article, prompt, require_dic):

    reward_score_all = 0.0

    # 重复指令
    prompt_to_repeat = require_dic.get("prompt_to_repeat", -1)
    if prompt_to_repeat != -1:
        s = "First repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)."
        prompt = prompt.replace(s, "").strip()
        if article.startswith(prompt):
            reward_score_all += 0.1
        article = article.replace(prompt, "").strip()

    # 高亮
    num_highlights = require_dic.get("num_highlights", -1)
    if num_highlights > 0:
        highlights = re.findall(r'(\*[^*]+\*|_[^_]+_)', article)
        if len(highlights) >= num_highlights:
            reward_score_all += 0.05

    # 文章单词数量
    low_word_cnt = require_dic.get("low_word_cnt", -1)
    max_word_cnt = require_dic.get("max_word_cnt", -1)
    words = re.findall(r'\b\w+\b', article)
    word_count = len(words)
    if low_word_cnt > 0 and max_word_cnt > low_word_cnt:
        if word_count >= low_word_cnt and word_count <= max_word_cnt:
            reward_score_all += 0.1
    elif low_word_cnt > 0:
        if word_count >= low_word_cnt:
            reward_score_all += 0.05
    elif low_word_cnt <= 0 and max_word_cnt > 0:
        if word_count <= max_word_cnt:
            reward_score_all += 0.05

    # 占位符数量
    num_placeholders = require_dic.get("num_placeholders", -1)
    if num_placeholders > 0:
        placeholders = re.findall(r'\[[^\[\]]+\]', article)
        placeholder_count = len(placeholders)
        if placeholder_count >= num_placeholders:
            reward_score_all += 0.05

    # 要点个数(这个和num_highlights只能2选1）,<=0表示没有要求
    num_bullets = require_dic.get("num_bullets", -1)
    if num_bullets > 0:
        bullet_points = [line for line in article.split('\n') if line.startswith('* ')]
        if len(bullet_points) == num_bullets:
            reward_score_all += 0.1

    # 全大写单词数量, 如果max_capital_cnt<=0，low_capital_cnt<=0，表示没有要求
    # low_capital_cnt<=0，表示没有最少
    # max_capital_cnt<=0，表示没有最多
    # low_capital_cnt > max_capital_cnt，表示没有最多
    # low_capital_cnt==max_capital_cnt，表示只能出现max_num次
    low_capital_cnt = require_dic.get("low_capital_cnt", -1)
    max_capital_cnt = require_dic.get("max_capital_cnt", -1)
    words = article.split()
    caps_count = sum(1 for word in words if word.isupper() and len(word) >= 2)
    if low_capital_cnt > 0 and max_capital_cnt > low_capital_cnt:
        if caps_count <= max_capital_cnt and caps_count >= low_capital_cnt:
            reward_score_all += 0.1
    elif low_capital_cnt > 0 and max_capital_cnt < low_capital_cnt:
        if caps_count >= low_capital_cnt:
            reward_score_all += 0.05
    elif low_capital_cnt > 0 and low_capital_cnt == max_capital_cnt:
        if caps_count == low_capital_cnt:
            reward_score_all += 0.1
    elif low_capital_cnt <= 0 and max_capital_cnt > 0: 
        if caps_count <= max_capital_cnt:
            reward_score_all += 0.05

    # 关键词, 如果所有的low_num<0，max_num<0，表示没有关键词要求
    # 如果low_num<0，表示没有最少多少字要求
    # 如果max_num<0，表示没有最多多少字要求
    # 如果low_num > max_num，表示没有最多多少字要求
    # 如果low_num==max_num，表示只能出现max_num次
    keywords = require_dic.get("keywords", [])
    keywords_num = require_dic.get("keywords_num", [])
    find_no_apper = []
    if keywords and keywords_num and len(keywords) == len(keywords_num):
        res_ = []
        for k, v in zip(keywords_num, keywords):
            low_num, max_num = k[0], k[1]
            cnt = article.count(v)
            if low_num > 0 and max_num > low_num:
                if cnt >= low_num and cnt <= max_num:
                    reward_score_all += 0.1
            elif low_num > 0 and max_num < low_num:
                if cnt >= low_num:
                    reward_score_all += 0.05
            elif low_num > 0 and low_num == max_num:
                if cnt == low_num:
                    reward_score_all += 0.1
            elif low_num <= 0 and max_num > 0:
                if cnt <= max_num:
                    reward_score_all += 0.05
            elif low_num <= 0 and max_num <= 0:
                if cnt == 0:
                    reward_score_all += 0.05

    # 句子个数，-1表示没有要求
    num_sentences = require_dic.get("num_sentences", -1)
    if num_sentences > 0:
        sentences_style = require_dic.get("sentences_style", -1)
        sentence_endings = re.compile(r'[.!?]')
        # 移除换行和多余空格后进行计数
        sentences = sentence_endings.split(article.strip())
        # 去除空白句子
        cnt = len([s for s in sentences if s.strip()])
        if sentences_style == -1:
            if cnt < num_sentences:
                reward_score_all += 0.05
        elif sentences_style == 0:
            if cnt == num_sentences:
                reward_score_all += 0.2
        elif sentences_style == 1:
            if cnt > num_sentences:
                reward_score_all += 0.05

    # 字母要求, low_letter_num>0: 至少出现多少次；max_letter_num>0: 最多出现多少次
    # max_letter_num<=0,不能出现
    low_letter_num = require_dic.get("low_letter_num", -1)
    max_letter_num = require_dic.get("max_letter_num", -100)
    letter = require_dic.get("letter", "")
    cnt = article.count(letter)
    if low_letter_num > 0 and letter != "":
        if cnt >= low_letter_num:
            reward_score_all += 0.05
    elif max_letter_num > 0 and letter != "":
        if cnt <= max_letter_num:
            reward_score_all += 0.05
    elif max_letter_num < 0 and max_letter_num != -100 and letter != "":
        if cnt == 0:
            reward_score_all += 0.2

    # 段落要求
    # 几个段落，-1表示没有要求
    paragraphs = require_dic.get("paragraphs", "")
    paragraphs_num = require_dic.get("paragraphs_num", -1)
    if paragraphs_num > 0:
        if paragraphs == "\n\n" or paragraphs == "\n\n\n" or paragraphs == "***":
            paragraphs_list = [k for k in article.split(paragraphs) if k.strip() != ""]
            cnt = len(paragraphs_list)
            if cnt == paragraphs_num:
                reward_score_all += 0.1

    # 结尾, 为空表示没有要求
    end_phrase = require_dic.get("end_phrase", "")
    if end_phrase != "":
        if article.endswith(end_phrase):
            reward_score_all += 0.1

    # 特定段落要求, 如果nth_paragraph>0, first_word作用在段落上
    nth_paragraph = require_dic.get("nth_paragraph", -1)
    if nth_paragraph > 0:
        first_word = require_dic.get("first_word", "")
        if first_word != "":
            paragraphs_list = [k for k in article.split(paragraphs) if k.strip() != ""]
            if len(paragraphs_list) >= nth_paragraph and (paragraphs_list[nth_paragraph-1].startswith(first_word)):
                reward_score_all += 0.1
            
    return reward_score_all

def cosine_length_reward(length=0, max_length=8192, correct_score=0, correct=0, 
                         rc_0=0.0, rc_L=1.0, rw_0=0.0, rw_L=-1.0, r_exceed=-2.0):
    """
    标量版本的余弦长度奖励函数

    参数：
        length: 当前生成的长度（int 或 float）
        max_length: 最大允许生成长度
        correct: 是否正确（1 表示正确，0 表示错误）
        rc_0, rc_L: 正确答案的最短/最长长度的奖励值
        rw_0, rw_L: 错误答案的最短/最长长度的惩罚值
        r_exceed: 超过最大长度的固定惩罚

    返回：
        奖励值（float）
    """
    if length >= max_length:
        return r_exceed

    # 计算余弦因子：0 到 1
    ratio = length / max_length
    cos_factor = 0.5 * (1 - math.cos(math.pi * ratio))

    if correct_score >= 0.2:
        reward = rc_0 + (correct_score * 2 - rc_0) * cos_factor
    else:
        reward = rw_0 + (rw_L - rw_0) * cos_factor

    return reward

def extract_scores(text):
    scores = re.findall(r'\[\[(\d+)\]\]', text)

    # 转为整数列表
    scores = [int(score) for score in scores if int(score) > 0 and int(score) <= 10]

    return scores[-1]

def compute_score(model_output: str, ground_truth: str, extra_info):
    ret_score = {
        "socre": 0.0,
        "response_length_score": 0.0,
        "point_score": 0.0,
        "acc_score": 0.0
    }
    if "</think>" not in model_output:
        return ret_score

    query = ""
    response_length = extra_info["valid_response_length"]
    answer = model_output.split("</think>")[-1].strip()

    if "check_code" in ground_truth:
        try:
            ground_truth = json.loads(ground_truth)
            ground_truth["messages"][1]["content"] = answer
            s = judge_answer(ground_truth)
            if s["check_code"] == True:
                ret_score["acc_score"] = 1.0

            query = ground_truth["messages"][0]["content"]
            prompt = prompt_jqzl_en(query, answer)
        except:
            return ret_score

        return ret_score

    ground_truth = json.loads(ground_truth)
    f = False
    try:
        if ground_truth.get("source", "") == "jqzl-en":
            ret_score["point_score"] = validate_article_step_score_en(answer, ground_truth["query"], ground_truth["jqzl_prompt"])
            f, other_info = validate_article_en(answer, ground_truth["query"], ground_truth["jqzl_prompt"])
        else:
            ret_score["point_score"] = validate_article_step_score(answer, ground_truth["jqzl_prompt"])
            f, other_info = validate_article(answer, ground_truth["jqzl_prompt"])

        prompt = prompt_jqzl_en(ground_truth["query"], answer)

        if f == True:
            ret_score["acc_score"] = 1.0
            length_reward = cosine_length_reward(response_length, 8192, ret_score["acc_score"] + ret_score["point_score"], 1)
        else:
            length_reward = cosine_length_reward(response_length, 8192, ret_score["acc_score"] + ret_score["point_score"], 0)
        ret_score["response_length_score"] = length_reward 
    except:
        return ret_score

    return ret_score

if __name__ == "__main__":
    print("OK")
