#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
import sys
import logging
from difflib import Differ
from time import time
import re
import os
from copy import deepcopy

import torch
from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.utils import import_user_module

# from flask import Flask, request, jsonify
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
Batch = namedtuple('Batch', 'ids src_tokens src_lengths, src_strs')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


class Args():
    def __init__(self, model_path="./model/ft_lang8_all_cged1620_cltc7_7/checkpoint3.pt:./model/ft_lang8_all_cged1620_cltc4_9/checkpoint3.pt:./model/ft_lang8_all_cged1620_cltc4_10/checkpoint3.pt", dict_path = ["./dicts"], raw_text=True,
                 batch_size=128, max_tokens=14464, max_len_a=0, max_len_b=113, beam_size=4, nbest=1, replace_unk=None,
                 copy_ext_dict=True, no_early_stop=False, print_alignment=False, no_progress_bar=True, max_len=112,
                 cpu=False, epoch=3,
                 seps_list=[r'''(?P<punc>([。！？!?]|……|\.{6})[”"]?|\.[”"])''', r'''(?P<punc>[；;])''', r'''(?P<punc>[：，、.,:])''']):
        self.batch_size = batch_size
        self.beam = beam_size
        self.copy_ext_dict = copy_ext_dict
        self.cpu = cpu
        self.data = dict_path
        self.diverse_beam_groups = -1
        self.diverse_beam_strength = 0.5
        self.fp16 = False
        self.fp16_init_scale = 128
        self.fp16_scale_tolerance = 0.0
        self.fp16_scale_window = None
        self.gen_subset = 'test'
        self.lazy_load = False
        self.left_pad_source = 'True'
        self.left_pad_target = 'False'
        self.lenpen = 1
        self.log_format = None
        self.log_interval = 1000
        self.match_source_len = False
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.max_sentences = batch_size
        self.max_source_positions = 113
        self.max_target_positions = 113
        self.max_tokens = max_tokens
        self.memory_efficient_fp16 = False
        self.min_len = 1
        self.min_loss_scale = 0.0001
        self.model_overrides = "{}"
        # self.model_overrides = "{'use_encoder_classification': False, 'use_sentence_copying': False}"
        self.nbest = nbest
        self.no_beamable_mm = False
        self.no_early_stop = no_early_stop
        self.no_progress_bar = no_progress_bar
        self.no_repeat_ngram_size = 0
        self.num_shards = 1
        self.num_workers = 0
        self.path = model_path
        self.prefix_size = 0
        self.print_alignment = print_alignment
        self.quiet = False
        self.raw_text = raw_text
        self.remove_bpe = None
        self.replace_unk = replace_unk
        self.sacrebleu = False
        self.sampling = False
        self.sampling_temperature = 1
        self.sampling_topk = -1
        self.score_reference = False
        self.seed = 1
        self.shard_id = 0
        self.skip_invalid_size_inputs_valid_test = False
        self.source_lang = "src"
        self.target_lang = "tgt"
        self.task = 'translation'
        self.tensorboard_logdir = ''
        self.threshold_loss_scale = None
        self.unkpen = 0
        self.unnormalized = False
        self.upsample_primary = 1
        self.user_dir = None

        self.split_level = 0
        self.seps = seps_list
        self.max_len = max_len

        self.epoch = epoch


class GECServer():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.use_cuda = torch.cuda.is_available() and not args.cpu
        self.epoch = 0

        # Setup task, e.g., translation
        self.task = tasks.setup_task(args)

        self.max_positions = None

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        self.align_dict = None

        # Initialize generator
        self.generator = self.task.build_generator(args)

        self.models = None

        self.diff = Differ()

    def load_model(self):
        task = self.task
        args = self.args

        # Load ensemble
        # self.logger.info('| loading model(s) from {}'.format(args.path))
        print('| loading model(s) from {}'.format(args.path))
        self.models, _model_args = utils.load_ensemble_for_inference(
            args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
        )

        # 模型有copy_attention，则copy_ext_dict为True
        # args.copy_ext_dict = getattr(_model_args, "copy_attention", False)

        # Optimize ensemble for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment or args.copy_ext_dict,
            )   # 生成时计算attention
            if args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(args.replace_unk)
        # if self.align_dict is None and args.copy_ext_dict:
        #     self.align_dict = {}

        self.max_positions = utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in self.models]
        )

    def make_batches(self, lines):
        task = self.task
        args = self.args

        tokens = [
            task.source_dictionary.encode_line(src_str, add_if_not_exist=False, copy_ext_dict=args.copy_ext_dict).long()
            for src_str in lines
        ]

        lengths = torch.LongTensor([t.numel() for t in tokens])
        itr = task.get_batch_iterator(
            dataset=task.build_dataset_for_inference(tokens, lengths),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=self.max_positions,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            yield Batch(
                ids=batch['id'],
                src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
                src_strs=[lines[i] for i in batch['id']],
            )

    def _convert_sep(self, sep):
        """
        将分句符转换为 sp + "|||"
        :param sep: 分句符
        :return:
        """
        sp = sep.group("punc")
        return sp + "|||"

    def split_art(self, art):
        """
        递归分句
        1.分句先按中文句号/中英问号/中英叹号/中文省略号……/英文....../前面所说分句符组合中英双引号/英文句号组合中英双引号，分句后句长<=112的不再分句
        2.对1中句长>112的句子，再按中英文分号分句，分句后句长<=112的不再分句
        3.对2中句长>112的句子，再按中文冒号、中文逗号、英文句号等其他标点符号分句，之后将长度<112的句子进行拼接且拼接后的句长<=112
        4.对3中句长>112的句子（句子中没有标点符号），暴力截取
        :param art:
        :return art_sents:
        """
        self.args.split_level += 1   # 进入下一级分句
        art_sents = []

        if self.args.split_level != 4:
            seps = self.args.seps[self.args.split_level - 1]
            art = re.sub(seps, self._convert_sep, art)
            sents = art.split("|||")
            if sents[-1] == "": # 去掉空句 2020.11.24
                sents = sents[:-1]

            concat_sent = ""
            for i, sent in enumerate(sents):
                if len(sent) <= self.args.max_len:
                    if self.args.split_level != 3:
                        art_sents.append(sent)
                    else:   # 拼接
                        if len(concat_sent + sent) <= self.args.max_len:
                            concat_sent += sent
                        else:
                            art_sents.append(concat_sent)
                            concat_sent = sent  # 缓存当前句
                        if i == len(sents) - 1:  # 保存最后一个<max_len的句子 2020.11.24
                            art_sents.append(concat_sent)
                else:
                    if self.args.split_level == 3 and concat_sent:
                        # 当前句子>max_len，拼接操作中断，保存前一句子，从下一句子重新开始拼接 2020.11.23
                        art_sents.append(concat_sent)
                        concat_sent = ""
                    sub_sents = self.split_art(sent)
                    art_sents.extend(sub_sents)
        else:   # 暴力截取
            len_art = len(art)
            max_splits = len_art // self.args.max_len
            for i in range(max_splits):
                art_sents.append(art[(i * self.args.max_len):((i + 1) * self.args.max_len)])
            if len_art % self.args.max_len != 0:
                art_sents.append(art[(max_splits * self.args.max_len):])

        # 返回上一级分句
        self.args.split_level -= 1

        return art_sents

    def correct_sents(self, texts):
        args = self.args

        # 切分文本。分句保持原样？
        regexp = r"\s|\\n|\\r"
        txt_id, sents = [], []
        for text in texts:
            # 样本中r"\n"、r"\r"实际代表1个字符
            # sts = self.split_art(re.sub(regexp, "я", text))
            text = re.sub(regexp, "я", text)
            sts = self.split_art(text) if len(text) > 112 else [text]
            # sts = [" ".join(st) + "я" for st in sts]   # 解决生成时的多余文本问题
            # 解决多余文本问题。请求文本字符之间不以空格分隔
            # 替换文本中的空白符为я，输入模型后变为unk，避免以空格分词产生的问题
            # sts = [" ".join(st.replace(" ", "я")) + "\n" for st in sts]
            # sts = [" ".join(st) for st in sts]
            sts = [" ".join(st.lower()) for st in sts]
            # sts = [st + "\n" for st in sts]
            t_id = txt_id[-1] + len(sts) if txt_id else len(sts)
            txt_id.append(t_id)
            sents.extend(sts)

        start_id = 0
        results = []
        corrects = []
        scores = []

        for batch in self.make_batches(sents):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths

            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }

            translations = self.task.inference_step(self.generator, self.models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            # if self.src_dict is not None:
            #     src_str = self.src_dict.string(src_tokens, args.remove_bpe)

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=sents[id],
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=args.remove_bpe,
                )

                regexp = r"<eos>|<unk>"
                # src_str, hypo_str = src_str.replace("<unk>", "|").replace(" ", ""), re.sub(regexp, "|", hypo_str).replace(" ", "")
                # cors = self.get_edits(src_str, hypo_str)
                # corrects.append(cors)
                hypo_str = re.sub(regexp, "|", hypo_str).replace(" ", "")

                token_scores = torch.exp(hypo["positional_scores"]).tolist()
                score = np.exp(hypo['score'])

                # print("score ", score)
                # scores.append(score)

                # 统计错误句子，正确句子，所有句子。分句后？
                # sent num, avg score, min score, max score  2467 0.908242720494238 0.6215933723683859 0.9953903153107126
                # if score < 0.9082:
                #     hypo_str = sents[id].replace(" ", "")

                # [err_type, err_start, err_end, err_str, cor_str, cor_start]
                # src_str = sents[id].replace(" ", "")
                # edits = self.get_edits(src_str, hypo_str)
                #
                # if edits:
                #     for i, edit in enumerate(deepcopy(edits)):
                #         edit_flag = True
                #         # 修正文本中的token有<阈值的时就过滤访修正
                #         # if edit[0] != "R" and any([True if token_scores[edit[-1] + j] < 0.6499 else False for j, c in enumerate(edit[4])]):
                #         if edit[0] != "R" and np.mean([token_scores[edit[-1] + j] for j, c in enumerate(edit[4])]) < 0.:#51875:
                #                 # or edit[0] == "R" and score < 0.89875:   # 0.58875 0.89875   # 0.59 0.55
                #             edit_flag = False
                #         edits[i][-1] = edit_flag

                        # sent num, avg score, min score, max score  2260 0.6499240523632781 0.0452645942568779 0.9999938607215881
                        # "O", err_str==cor_str？
                        # if edit[0] != "R":
                        #     scores.extend([token_scores[edit[-1] + j] for j, c in enumerate(edit[4])])

                # 调试token_scores bug
                # edits_num = len(edits)
                # if edits_num > 0:
                #     # try:
                #     i = 0
                #     while i < edits_num:
                #         edit_flag = True
                #         if edits[i][0] != "R" and any(
                #                 [True if token_scores[edits[i][-1] + j] < 0 else False for j, c in enumerate(edits[i][4])]):
                #             edit_flag = False
                #         # if edits[i][0] != "R":
                #         #     scores.extend([token_scores[edits[i][-1] + j] for j, c in enumerate(edits[i][4])])
                #         edits[i][-1] = edit_flag
                #         i += 1
                #     # except Exception as e:
                #     #     print("src_str", src_str)
                #     #     print("edits", edits)
                #     #     print("token_scores", token_scores)
                #     #     print(e)
                #     #     exit(-1)

                    # hypo_str = src_str
                    # # edits merge, 无错/有错（纠对）token-->edit阈值，token edit flag-->get_edits()
                    # edits = [edit[:-1] for edit in edits if edit[-1]]
                    # if edits and len(src_str) >= 8: # <8不纠错
                    #     for edit in edits[::-1]:
                    #         if edit[0] != "R":
                    #             hypo_str = hypo_str[:edit[1]] + edit[4] + hypo_str[edit[2]:]
                    #         else:
                    #             hypo_str = hypo_str[:edit[1]] + hypo_str[edit[2]:]

                corrects.append(hypo_str)

        # print("sent num, avg score, min score, max score ", len(scores), np.mean(scores), np.min(scores), np.max(scores))
        # exit(-1)

        # 合并句子以还原文本
        corrects = ["".join(corrects[txt_id[i-1]:t_id]) if i > 0 else "".join(corrects[:t_id]) for i, t_id in enumerate(txt_id)]

        # 舍弃非中文、纠错前后长度差大于8的纠错，修改原句
        # [err_type, err_start, err_end, err_str, cor_str, cor_start]
        assert len(texts) == len(corrects)
        for i, txt in enumerate(texts):
            edits = self.get_edits(txt, corrects[i])
            if edits and len(txt) >= 8: # <8不纠错
                for edit in edits[::-1]:
                    if edit[0] != "R":
                        txt = txt[:edit[1]] + edit[4] + txt[edit[2]:]
                    else:
                        txt = txt[:edit[1]] + txt[edit[2]:]

            if self.args.epoch == 1:
                corrects[i] = " ".join(txt)
            else:
                corrects[i] = txt

        self.epoch += 1
        if self.epoch == self.args.epoch:
            if self.args.epoch == 1:
                self.epoch -= 1
            return corrects

        # 多轮纠错，最终纠错结果和起始输入比对；当次纠错结果和输入比对
        corrects = self.correct_sents(corrects)
        self.epoch -= 1

        if self.epoch == 1:
            # 舍弃非中文、纠错前后长度差大于8的纠错，修改原句
            # [err_type, err_start, err_end, err_str, cor_str, cor_start]
            assert len(texts) == len(corrects)
            for i, txt in enumerate(texts):
                edits = self.get_edits(txt, corrects[i])
                if edits and len(txt) >= 8:  # <8不纠错
                    for edit in edits[::-1]:
                        if edit[0] != "R":
                            txt = txt[:edit[1]] + edit[4] + txt[edit[2]:]
                        else:
                            txt = txt[:edit[1]] + txt[edit[2]:]
                corrects[i] = " ".join(txt)

            self.epoch -= 1

        return corrects

    def get_edits(self, src_str, pred_str):
        """
        添加edits_offset
        :param src_str:
        :param pred_str:
        :return:
        """
        diff = self.diff.compare(src_str, pred_str)
        edits = []
        src_i = 0  # src_str中可能的下一个错误位置
        err_start, err_end = 0, 0
        err_str, cor_str = "", ""
        diff = list(diff)
        diff_len = len(diff)
        # 添加字词，位置后移；删除字词，位置前移。乱序错误，添加、删除字词而前后移动相互抵消
        edits_offset = 0
        for i, df in enumerate(diff):
            # 不同类型的错误及其组合的纠正，都可以转化为去掉只属于src_str的字或添加只属于pred_str的字
            # 组合错误的纠正，去掉只属于src_str的连续的字，添加只属于pred_str的连续的字
            # eg: ['  我', '- 喜', '- 欢', '+ 西', '+ 环', '  吃', '  红', '  烧', '  肉']
            if df[0] == "-":
                # 多字
                err_str += df[2]
                src_i += 1  # src_str的下一个字符可能无错
                err_end = src_i
            elif df[0] == "+":
                # 少字，err_end不变
                cor_str += df[2]

            # 错误结束或没有错误，或者最后一个包含错误
            if (df[0] in [" ", "?"]) or (i == diff_len - 1):
                # 检测到错误
                if err_str or cor_str:
                    len_err, len_cor = len(err_str), len(cor_str)
                    if len_err == 0 and len_cor > 0:
                        err_type = "M"
                    elif len_err > 0 and len_cor > 0:
                        err_type = "S"
                    else:
                        err_type = "R"

                    # 如果之前为少字错误而当前为多字错误或之前为多字错误而当前为少字错误，且多的字和少的字相等，则合并为顺序颠倒错误
                    # err_str或cor_str包含"|"时，该顺序颠倒无错
                    # 之前已检测到错误
                    if edits:
                        pre_err_type, pre_err_start, pre_err_end, pre_err_str, pre_cor_str, _ = edits[-1]
                        if pre_err_type == "M" and err_type == "R" and pre_cor_str.replace("|", "") == err_str.replace("|", "") \
                                or pre_err_type == "R" and err_type == "M" and pre_err_str.replace("|", "") == cor_str.replace("|", ""):
                            edits_offset += (len(cor_str) - len(err_str))
                            edits.pop()
                            err_str = src_str[pre_err_start:err_end]
                            # 之前为少字错误而当前为多字错误
                            if pre_err_type == "M" and err_type == "R":
                                cor_str = pre_cor_str + src_str[pre_err_end:err_start]
                            else:
                                cor_str = src_str[pre_err_end:err_start] + cor_str
                            err_start = pre_err_start
                            err_type = "O"

                    def is_chinese(string):
                        if string == "":
                            chinese_flag = True
                        else:
                            chinese_flag = False
                            for c in string:
                                if '\u4e00' <= c <= '\u9fa5':
                                    chinese_flag = True
                                else:
                                    chinese_flag = False
                                    break
                        return chinese_flag

                    # 不在句末纠错，一般为标点符号；过滤纠错长度>4
                    # abs(len(err_str) - len(cor_str)) > 8
                    # if not is_chinese(err_str) or not is_chinese(cor_str) or len(err_str) > 4 or len(cor_str) > 4 or err_start == (len(src_str) - 1):
                    #     err_type = "C"
                    if err_type != "C":
                        # edits.append([err_type, err_start, err_end, err_str, cor_str])
                        edits.append([err_type, err_start, err_end, err_str, cor_str, err_start + edits_offset])
                        edits_offset += (len(cor_str) - len(err_str))
                    err_str, cor_str = "", ""

                # 动态更新src_str中可能的下一个错误位置
                src_i += 1
                err_end = src_i
                err_start = src_i

        return edits

    def filter_edits(self, src_sents, pred_sents):
        assert len(src_sents) == len(pred_sents)
        for i, txt in enumerate(src_sents):
            edits = self.get_edits(txt, pred_sents[i])
            if edits and len(txt) >= 8:  # <8不纠错
                for edit in edits[::-1]:
                    if edit[0] != "R":
                        txt = txt[:edit[1]] + edit[4] + txt[edit[2]:]
                    else:
                        txt = txt[:edit[1]] + txt[edit[2]:]
            pred_sents[i] = txt
        return pred_sents


logging.basicConfig(filename='./log/server.log', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

args = Args()
gec_server = GECServer(args, logger)
gec_server.load_model()

# app = Flask(__name__)
#
# @app.route('/correct', methods=['POST'])
# def gec():
#     try:
#         data = request.data.decode('utf-8')
#     except Exception as e:
#         logger.info(e)
#         data = None
#
#     if data:
#         texts = data.split('|||')
#         corrects = gec_server.correct_sents(texts)
#         results = jsonify(corrects=corrects)
#     else:
#         results = bad_request()
#     return results
#
# @app.errorhandler(400)
# def bad_request():
#     message = {
#         'status': 400,
#         'message': 'Bad data',
#     }
#     resp = jsonify(message)
#     resp.status_code = 400
#     return resp

def local_infer(bs=6400):
    # f_path = r"./data/benchmark"
    f_path = r"./data/cltc/test"
    # 预测数据文件名的后半部分
    # test_lable = "ft_dae2_5_lb_plw2_5_4.txt"
    test_lable = "ft455_lang8_all_cged1620_cltc744_cged_test.txt"
    # test_lable = "test.txt"
    # 测试数据文件名的开始部分
    # file_pre = "bm_all_norm"
    # file_pre = "ctc"
    # file_pre = "sc"
    # file_pre = "sighan"
    # file_pre = "bm_all_norm"
    # file_pre = "site_collection_split"
    file_pre = "cged_test"

    regexp = r"\s|\\n|\\r"
    # file = os.path.join(f_path, file_pre + ".src-tgt.src")
    file = os.path.join(f_path, file_pre + ".src")
    with open(file, "r", encoding="utf-8") as f:
        src_txts = f.readlines()

        # 微调数据
        # src_txts = [src_txt.rstrip("\n").replace(" ", "") for src_txt in src_txts]

        # txts = list(src_txts)

        # src_txts = [re.sub(regexp, "я", src_txt.rstrip("\n")) for src_txt in src_txts]

        # src_txts = [src_txt[10:].rstrip("\n") for src_txt in src_txts]

        src_txts = [src_txt.strip().split("\t")[1] for src_txt in src_txts]

    file_pre = "cged_test_bert"
    file = os.path.join(f_path, file_pre + ".src")
    with open(file, "r", encoding="utf-8") as f:
        bert_preds = f.readlines()
        bert_preds = [pred.strip().split("\t")[1] for pred in bert_preds]

    # src_txts = gec_server.filter_edits(src_txts, bert_preds)

    t_start = time()
    pred_txts = []
    for i in range(0, len(src_txts), bs):
        preds = gec_server.correct_sents(src_txts[i:(i+bs)])

        # preds = [txts[j][:10] + pd for j, pd in enumerate(preds)]

        pred_txts.extend(preds)
    print("纠错耗时", time() - t_start)
    # pred_txts = [" ".join(pred_txt) + "\n" for pred_txt in pred_txts]
    pred_txts = [pred_txt + "\n" for pred_txt in pred_txts]

    file = os.path.join(f_path, "predict_result_" + test_lable)
    with open(file, "w", encoding="utf-8") as f:
        f.writelines(pred_txts)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5201)
    local_infer()
