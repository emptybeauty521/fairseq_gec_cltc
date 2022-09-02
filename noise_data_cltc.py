# -*- coding: utf-8 -*-
"""
生成伪纠错数据

hanlp分词简介：https://www.cnblogs.com/adnb34g/p/10231700.html
"""

import hashlib
import re
import os
import json

from tqdm import tqdm
import numpy as np
from pyhanlp import HanLP


class NoiseInjector():
    def __init__(self, conf, tokenizer, synonyms, char_pron_conf=None, char_shape_conf=None, word_pron_conf=None, freq_char=None, freq_word=None):
        """
        :param conf: 字词加权混淆集
        :param char_pron_conf: 字音混淆集
        :param char_shape_conf: 字形混淆集
        :param word_pron_conf: 词音混淆集
        :param tokenizer: 分词器
        :param freq_char: 高频字
        :param freq_word: 高频词
        :param synonyms: 词语的近义词
        """
        # self.char_pron_conf = char_pron_conf
        # self.char_shape_conf = char_shape_conf
        self.conf = conf
        self.freq_char = freq_char
        self.freq_word = freq_word
        # self.word_pron_conf = word_pron_conf
        self.synonyms = synonyms
        self.tokenizer = tokenizer
        self.replace_a, self.replace_b = None, None
        self.delete_a, self.delete_b = None, None
        self.add_a, self.add_b = None, None
        self.swap_a, self.swap_b = None, None
        self.opt_level = None   # 操作级别，CHAR字级，WORD词级
        self.err_num = 0    # 句子的造错数量
        self.err_stat = {"char_level": [], "word_level": []}    # 造错信息统计
        self.replace_mean, self.delete_mean, self.add_mean, self.swap_mean = 0.1, 0.1, 0.1, 0.1 # 替换、删除、插入、交换造错概率

    def init_opt_prob(self, opt_level, replace_mean=0.1, delete_mean=0.1, add_mean=0.1, swap_mean=0.05,
                      replace_std=0.03, delete_std=0.03, add_std=0.03, swap_std=0.03):
        """初始化操作概率"""
        self.opt_level = opt_level
        self.replace_a, self.replace_b = self._solve_ab_given_mean_var(replace_mean, replace_std ** 2)
        self.delete_a, self.delete_b = self._solve_ab_given_mean_var(delete_mean, delete_std ** 2)
        self.add_a, self.add_b = self._solve_ab_given_mean_var(add_mean, add_std ** 2)
        self.swap_a, self.swap_b = self._solve_ab_given_mean_var(swap_mean, swap_std ** 2)
        # self.replace_mean, self.delete_mean, self.add_mean, self.swap_mean = replace_mean, delete_mean, add_mean, swap_mean

    @staticmethod
    def _solve_ab_given_mean_var(mean, var):
        a = mean * mean * (1. - mean) / var - mean
        b = (1. - mean) * (mean * (1. - mean) / var - 1.)
        return a, b

    def _swap_func(self, tgt, radius=3):
        """
        对句子进行交换造错
        :param radius: 将需要造错的字词平移到其邻域为radius的某个位置
        """
        # 交换造错概率
        swap_ratio = np.random.beta(self.swap_a, self.swap_b)
        rnd = np.random.random(len(tgt))

        # 将需要造错的字词平移到其邻域为radius的某个位置
        i = 0
        while i < len(tgt):
            w = tgt[i]
            if w[1] == 1 and rnd[i] < swap_ratio:
                w[1] = 0

                # 字在词内平移，确定词的范围；词在句中平移
                if self.opt_level == "CHAR":
                    if w[3] == 1:
                        # 该字为词尾
                        end = i + 1
                    else:
                        for j, c in enumerate(tgt[(i+1):]):
                            if c[3] == 1:
                                end = (i + 1 + j) + 1
                                break
                    if i == 0:
                        # 该字在句首
                        start = i
                    else:
                        tmp = tgt[:i]
                        tmp.reverse()
                        start = -1
                        for j, c in enumerate(tmp):
                            if c[3] == 1:
                                start = i - j
                                break
                        # 当前词为句中首词
                        if start == -1:
                            start = 0
                elif self.opt_level == "WORD":
                    start, end = 0, len(tgt)

                # 不对单字词、单词句造错
                if end - start > 1:
                    # 随机选择需要造错的字词其邻域为radius的某个位置
                    before_scope = np.arange(max(start, i-radius), i)
                    after_scope = np.arange(i+1, min(i+1+radius, end))
                    scope = np.concatenate((before_scope, after_scope))
                    idx = int(np.random.choice(scope))

                    # 不在非原句中的词语、非中文词语之间平移
                    if not (self.opt_level == "WORD" and (not self.is_chinese(tgt[idx][0]) or tgt[idx][2] == -1)):
                        # 该字为词尾时，将其前1个字变为词尾
                        # 移动到的位置的字为词尾时，将当前字变为词尾
                        if self.opt_level == "CHAR":
                            if w[3] == 1:
                                tgt[i-1][3] = 1
                            if tgt[idx][3] == 1:
                                w[3] = 1

                        # edit = ["O", w[0], tgt[idx][0]]
                        # if self.opt_level == "CHAR":
                        #     self.err_stat["char_level"].append(edit)
                        # else:
                        #     self.err_stat["word_level"].append(edit)

                        tgt.pop(i)
                        tgt.insert(idx, w)

                        self.err_num += 1

            i += 1

        return tgt

    def _replace_func(self, tgt):
        """
        对句子进行替换造错
        """
        # 替换造错概率
        ret = []
        replace_ratio = np.random.beta(self.replace_a, self.replace_b)
        rnd = np.random.random(len(tgt))

        # 对句子进行替换造错
        # 字级别，0.9的概率进行混淆集替换，0.1的概率进行近义词替换；词级别，0.1的概率进行近义词替换，0.5的概率进行混淆集替换，0.4的概率随机选择高频词进行替换
        for i, w in enumerate(tgt):
            word = w[0]
            if w[1] == 1 and rnd[i] < replace_ratio:
                sample = np.random.random_sample()
                if self.opt_level == "CHAR":
                    if sample <= 0.9:
                        conf = self.conf[word] if self.conf.get(word) else []
                    else:
                        conf = self.synonyms[word] if self.synonyms.get(word) else []
                else:
                    if sample <= 0.1:
                        conf = self.synonyms[word] if self.synonyms.get(word) else []
                    elif sample <= 0.6:
                        conf = self.conf[word] if self.conf.get(word) else []
                    else:
                        conf = self.freq_word

                # 有混淆集或近义词时，从其中随机选择一个字词进行替换
                # [rnd_word, 0（不再对该字或词造错）, -1（非句中原本的字或词）]
                # 字级造错再加一个词尾标志（1词语的最后一个字，0非词语的最后一个字）
                if conf:
                    rnd_word = np.random.choice(conf)
                    repl_w = [rnd_word, 0, -1]
                    if self.opt_level == "CHAR":
                        repl_w.append(1)
                    ret.append(repl_w)

                    self.err_num += 1

                    # edit = ["S", rnd_word, w[0]]
                    # if self.opt_level == "CHAR":
                    #     self.err_stat["char_level"].append(edit)
                    # else:
                    #     self.err_stat["word_level"].append(edit)
                else:
                    ret.append(w)
            else:
                ret.append(w)

        return ret

    def _delete_func(self, tgt):
        """
        对句子进行删除造错
        """
        # 删除句中的字词，只对长度不大于4的字词
        delete_ratio = np.random.beta(self.delete_a, self.delete_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, w in enumerate(tgt):
            if w[1] == 1 and rnd[i] < delete_ratio and len(w[0]) <= 4:
                # 删除的字为词尾时，将其前1个字变为词尾
                if self.opt_level == "CHAR" and i > 0 and w[3] == 1:
                    ret[-1][3] = 1

                self.err_num += 1

                # edit = ["M", "", w[0]]
                # if self.opt_level == "CHAR":
                #     self.err_stat["char_level"].append(edit)
                # else:
                #     self.err_stat["word_level"].append(edit)

                continue

            ret.append(w)

        return ret

    def _add_func(self, tgt):
        """
        对句子进行插入造错
        """
        # 高频字、高频词、近义词
        if self.opt_level == "CHAR":
            vocab = self.freq_char
        elif self.opt_level == "WORD":
            vocab = self.freq_word
        synonyms = self.synonyms

        # 插入原句中的长度小于5的中文字词
        tokens = [w[0] for w in tgt if self.is_chinese(w[0]) and w[2] != -1 and len(w[0]) < 5]

        # 插入概率
        add_ratio = np.random.beta(self.add_a, self.add_b)
        ret = []
        rnd = np.random.random(len(tgt))

        # 对句子进行插入造错，之后不再对该字词进行其他造错
        # 0.1的概率插入"的"；0.2的概率插入近义词和当前字词，近义词和当前字词插入概率各半；0.7的概率插入高频字词和原句中字词，高频字词和原句中字词插入概率各半
        for i, w in enumerate(tgt):
            if w[1] == 1 and rnd[i] < add_ratio:
                w[1] = 0
                word = w[0]
                sample = np.random.random_sample()
                if sample <= 0.1:
                    rnd_word = "的"
                elif sample <= 0.2:
                    if synonyms.get(word):
                        synonym = np.random.choice(synonyms[word])
                        rnd_word = np.random.choice([word, synonym])
                    else:
                        rnd_word = word
                else:
                    if np.random.random_sample() <= 0.5 and tokens:
                        rnd_word = np.random.choice(tokens)
                    else:
                        rnd_word = np.random.choice(vocab)

                # [rnd_word, 0（不再对该字或词造错）, -1（非句中原本的字或词）]
                # 字级造错再加一个词尾标志（1词语的最后一个字，0非词语的最后一个字）
                insert_w = [rnd_word, 0, -1]
                if self.opt_level == "CHAR":
                    insert_w.append(1)

                # 插入到当前字词前或后的概率各半
                if np.random.random_sample() > 0.5:
                    ret.append(insert_w)
                    ret.append(w)
                    # err_w = rnd_word + w[0]
                else:
                    ret.append(w)
                    ret.append(insert_w)
                    # err_w = w[0] + rnd_word

                self.err_num += 1

                # edit = ["R", err_w, w[0]]
                # if self.opt_level == "CHAR":
                #     self.err_stat["char_level"].append(edit)
                # else:
                #     self.err_stat["word_level"].append(edit)
            else:
                ret.append(w)

        return ret

    def parse(self, pairs):
        """
        获取造错句子、造错句子与源句间的对齐信息
        """
        align = []
        art = []
        for si in range(len(pairs)):
            ti = pairs[si][2]
            w = pairs[si][0]
            art.append(w)
            if ti >= 0:
                align.append('{}-{}'.format(si, ti))
        return art, align

    def is_chinese(self, string):
        """
        判断字符串中的字符是否都为汉字
        :return: 字符串中的字符都为汉字则为True，否则为False
        """
        chinese_flag = False
        for c in string:
            if '\u4e00' <= c <= '\u9fa5':
                chinese_flag = True
            else:
                chinese_flag = False
                break
        return chinese_flag

    def inject_noise(self, tgt, seg=9):
        """
        在源句子上造错，以词语为单位，对词语只进行词级或字级的替换、删除、插入、交换中的一种造错操作
        :param tgt:源句
        :param seg:每seg个词语片段造错一次
        :return:造错句子，对齐信息，句中造错数量
        """
        # 分词
        tokens = self.tokenizer.seg(tgt)

        # 获取需要造错的词语
        # 仅在非人名的中文词语上造错
        noise_idx = []
        tokens_focus = []
        for i, t in enumerate(tokens):
            # 词语，词性
            w = str(t.word)
            n = str(t.nature)
            if self.is_chinese(w) and (n not in ["nr", "nr1", "nr2", "nrf", "nrj"]):
                noise_idx.append(i)
            tokens_focus.append(w)

        # 每seg个词的片段随机挑选1个位置植入错误
        noise_idx = [np.random.choice(noise_idx[i:(i+seg)]) for i in range(0, len(noise_idx), seg)]
        tokens_focus = [[w, 1] if i in noise_idx else [w, 0] for i, w in enumerate(tokens_focus)]

        # 将字符级的字符对齐转换为词级的字符对齐
        # 元素为[字, 造错标志（1造错，0不造）, 词的索引]
        pairs = []
        for i, token in enumerate(tokens_focus):
            if i == 0:
                idx_rel = 0
            else:
                # 前1单词首字符的索引+前1单词的长度
                idx_rel = pairs[i-1][2] + len(pairs[i-1][0])
            pairs.append(token + [idx_rel])

        # 词级造错
        # 清零上一句子的造错计数
        # 设置词级替换、删除、插入、交换造错的概率
        # self.err_stat = {"char_level": [], "word_level": []}
        self.err_num = 0
        funcs = [self._replace_func, self._delete_func, self._add_func, self._swap_func]
        np.random.shuffle(funcs)
        self.init_opt_prob(opt_level="WORD", replace_mean=0.12, delete_mean=0.07, add_mean=0.05, swap_mean=0.02)
        for f in funcs:
            pairs = f(pairs)

        # self.err_stat["word_err"] = self.err_num

        # 随机选择关注词中的一个字来造错
        # 将词级的字符对齐转换为字符级的字符对齐
        # [字, 造错标志（1造错，0不造，-1忽略此标志）, 字的索引（-1替换或插入的词）, 词尾标志（1词语的最后一个字，0非词语的最后一个字，-1忽略此标志）]
        pairs_new = []
        for pair in pairs:
            w_len = len(pair[0])
            if pair[1] == 1:
                err_ixd = np.random.choice(w_len)
                pair = [[char, 1, pair[2] + i, 0 if (i+1) < w_len else 1] if i == err_ixd else [char, 0, pair[2] + i, 0 if (i+1) < w_len else 1] for i, char in enumerate(pair[0])]
            else:
                pair = [[char, 0, pair[2] + i, 0 if (i+1) < w_len else 1] if pair[2] != -1 else [char, 0, -1, 0 if (i+1) < w_len else 1] for i, char in enumerate(pair[0])]
            pairs_new.extend(pair)

        # 字级造错
        # 设置字级替换、删除、插入、交换造错的概率
        np.random.shuffle(funcs)
        self.init_opt_prob(opt_level="CHAR", replace_mean=0.36, delete_mean=0.20, add_mean=0.16, swap_mean=0.02)
        for f in funcs:
            pairs_new = f(pairs_new)

        # self.err_stat["char_err"] = self.err_num - self.err_stat["word_err"]
        # self.err_stat["err_num"] = self.err_num

        src, align = self.parse(pairs_new)

        return src, align, self.err_num


def save_file(filename, contents):
    # 空格分隔字符
    with open(filename, "w", encoding="utf-8") as ofile:
        for content in contents:
            content = " ".join(content)
            ofile.write(content + "\n")


def noise(cor_file, conf_file, ofile_suffix, synonyms_file, char_pron_conf_file=None, char_shape_conf_file=None,
          word_pron_conf_file=None, freq_char_file=None, freq_word_file=None):
    """
    构造伪纠错数据
    :param cor_file: 源数据
    :param conf_file: 字词加权混淆集
    :param char_pron_conf_file: 字音混淆集
    :param char_shape_conf_file: 字形混淆集
    :param word_pron_conf_file: 词音混淆集
    :param ofile_suffix: 伪纠错数据保存路径
    :param freq_word_file: 关注词
    :param freq_char_file: 高频字
    :param synonyms_file: 词语的近义词
    :return:
    """
    # 根据源数据划分，获取划分数据
    with open(cor_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        arts = [line.rstrip("\n") for line in lines[((args.split-1)*args.num_per_split):(args.split*args.num_per_split)]]

    # 字词加权混淆集
    with open(conf_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.rstrip("\n").split("\t") for line in lines]
        conf = {line[0]: json.loads(line[1]) for line in lines}
    for w in conf:
        all_cf = []
        for cf in conf[w]:
            all_cf.extend([cf] * conf[w][cf])
        conf[w] = all_cf

    # 高频字、高频词
    with open(freq_char_file, "r", encoding="utf-8") as f:
        lines = f.read()
        freq_char = lines.rstrip("\n").split("\n")
    with open(freq_word_file, "r", encoding="utf-8") as f:
        lines = f.read()
        freq_word = lines.rstrip("\n").split("\n")

    # 近义词字典。
    # 选择长度不大于4的词语；词语长度为1时，选择长度不大于2的近义词；词语长度大于1而不大于4时，选择长度不大于4的近义词。
    # 过滤没有近义词的词语
    with open(synonyms_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.rstrip("\n").split() for line in lines]
        synonyms = {line[0]: line[1:] for line in lines}
        synonyms = {word: [s for s in ss if (len(word) == 1 and len(s) <= 2) or (len(word) > 1 and len(s) <= 4)]
                    for word, ss in synonyms.items() if len(word) <= 4}
        synonyms = {word: ss for word, ss in synonyms.items() if ss}

    # 维特比分词，使用自定义词典
    tokenizer = HanLP.newSegment('viterbi').enableCustomDictionary(True)
    noise_injector = NoiseInjector(conf, tokenizer, synonyms, freq_char=freq_char, freq_word=freq_word)

    # 生成伪纠错数据，对所有源数据造错，只保存错误数据
    # 按文章分句，每行一篇文章，文章的句子间以|||分隔
    # 空白符、换行、回车替换为я，大写字母转小写
    # md5去重
    srcs_err, tgts_err, cor_sents = [], [], []
    err_aligns, cor_aligns = [], []
    md5_sents = {}
    pbar = tqdm(arts)
    pbar.set_description("生成伪纠错数据：")
    err_n, char_err = [], []
    regexp = r"\s|\\n|\\r"
    for art in pbar:
        tgt_sents = art.split("|||")
        rnd = np.random.random(len(tgt_sents))
        for i, tgt in enumerate(tgt_sents):
            tgt = re.sub(regexp, "я", tgt).lower()
            md5_tgt = hashlib.md5(tgt.replace("я", "").encode("utf-8")).hexdigest()
            if not md5_sents.get(md5_tgt):
                if rnd[i] > 0.0:
                    src, align, err_num = noise_injector.inject_noise(tgt)
                    if err_num > 0:
                        srcs_err.append(src)
                        tgts_err.append(tgt)
                        err_aligns.append(align)

                        # err_n.append(err_num)
                        # char_err.append(err_stat["char_err"])
                        # print("".join(src))
                        # print(tgt)
                        # print(err_stat)
                    # else:
                    #     # 正确句子，质量不好
                    #     srcs_cor.append(tgt)
                    #     aligns_cor.append(align)
                else:
                    # 正确句子，高质量

                    # [字, 造错标志（1造错，0不造，-1忽略此标志）, 字的索引（-1替换或插入的词）, 词尾标志（1词语的最后一个字，0非词语的最后一个字，-1忽略此标志）]
                    pairs = [[w, -1, i, -1] for i, w in enumerate(tgt)]

                    # 生成对齐信息
                    _, align = noise_injector.parse(pairs)

                    cor_sents.append(tgt)
                    cor_aligns.append(align)

                md5_sents[md5_tgt] = 1

    # print(np.min(err_n), np.max(err_n), np.mean(err_n))
    # print("char_err, err_n ", np.sum(char_err), np.sum(err_n))

    # 随机化
    err_data = list(zip(srcs_err, tgts_err, err_aligns))
    np.random.shuffle(err_data)
    srcs_err, tgts_err, err_aligns = zip(*err_data)
    # nor_data = list(zip(cor_sents, cor_aligns))
    # np.random.shuffle(nor_data)
    # cor_sents, cor_aligns = zip(*nor_data)

    # 保存数据
    save_file('{}.err.src'.format(ofile_suffix), srcs_err)
    save_file('{}.err.tgt'.format(ofile_suffix), tgts_err)
    save_file('{}.err.forward'.format(ofile_suffix), err_aligns)
    # save_file('{}.nor.src'.format(ofile_suffix), cor_sents)
    # save_file('{}.nor.forward'.format(ofile_suffix), cor_aligns)

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=1, help="数据的划分数量")
parser.add_argument('--num_per_split', type=int, default=5000000, help="每份数据的数量")
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--data', type=str, default="arts_sents.txt")

args = parser.parse_args()
np.random.seed(args.seed)

if __name__ == '__main__':
    print("split={}, num_per_split={}, seed={}".format(args.split, args.num_per_split, args.seed))

    # 单语语料
    file_prefix = "./data/cltc/pretrain"
    cor_file = os.path.join(file_prefix, args.data)

    # 加权混淆集、高频字、高频词、近义词、数据保存路径
    conf_file = os.path.join(file_prefix, "conf_weighted.txt")
    freq_char_file = os.path.join(file_prefix, "freq_char.txt")
    freq_word_file = os.path.join(file_prefix, "freq_word.txt")
    synonyms_file = os.path.join(file_prefix, "synonyms.txt")
    ofile_suffix = './data/cltc/pretrain/sec{}'.format(args.seed)

    # 构造伪纠错数据
    noise(cor_file, conf_file, ofile_suffix, synonyms_file, freq_char_file=freq_char_file, freq_word_file=freq_word_file)
