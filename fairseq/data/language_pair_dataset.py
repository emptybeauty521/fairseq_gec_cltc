# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
修改：
添加 sentence-level copying task:
    collate(): err_flag
    get_err_labels()
    err_label
    ordered_indices()
"""

import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)
    
    src_label, tgt_label = None, None
    if samples[0].get('source_label', None) is not None:
        src_label = merge('source_label', left_pad=left_pad_target)
        src_label = src_label.index_select(0, sort_order)
    if samples[0].get('target_label', None) is not None:
        tgt_label = merge('target_label', left_pad=left_pad_target)
        tgt_label = tgt_label.index_select(0, sort_order)

    # batch_size中的样本包含错误时，err_flag为1，否则为0
    # dummy/oom batch没有'err_label'属性
    err_flag = False
    # if src_label is not None and tgt_label is not None:
    if samples[0].get('err_label', None) is not None:
        err_labels = np.array([s.get('err_label') for s in samples])
        if (err_labels == 1).any():
            err_flag = True

    # batch = {
    #     'id': id,
    #     'nsentences': len(samples),
    #     'ntokens': ntokens,
    #     'net_input': {
    #         'src_tokens': src_tokens,
    #         'src_lengths': src_lengths,
    #     },
    #     'target': target,
    #     'source_label': src_label,
    #     'target_label': tgt_label,
    # }
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'err_flag': err_flag,
        },
        'target': target,
        'source_label': src_label,
        'target_label': tgt_label,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict, src_label,
        tgt=None, tgt_sizes=None, tgt_dict=None, tgt_label=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
        err_cor_batch=False
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_label = src_label
        self.tgt_label = tgt_label
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.err_labels = self.get_err_labels() if err_cor_batch else None

    def get_err_labels(self):
        """
        获得样本的标签，0 不包含错误，1 包含错误
        """
        err_labels = None
        if self.src_label and self.tgt_label is not None:
            src_labels = self.src_label.labels_list
            tgt_labels = self.tgt_label.labels_list
            assert len(src_labels) == len(tgt_labels)
            err_labels = [0 if src_label.size()[0] == tgt_labels[i].size()[0] and (src_label == 0).all() and
                                    (tgt_labels[i] == 0).all() else 1 for i, src_label in enumerate(src_labels)]
            err_labels = np.array(err_labels)
        return err_labels

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        src_label_item = self.src_label[index] if self.src_label is not None else None
        tgt_label_item = self.tgt_label[index] if self.tgt_label is not None else None
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        # ！！！tgt添加了eos token及其label；加载数据时添加了eos label
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])
                if tgt_label_item is not None:
                    tgt_label_item = torch.cat([self.tgt_label[index], torch.IntTensor([0])])

        # 加载数据时src添加了eos label；src默认添加了eos token
        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        # return {
        #     'id': index,
        #     'source': src_item,
        #     'target': tgt_item,
        #     'source_label': src_label_item,
        #     'target_label': tgt_label_item,
        # }
        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'source_label': src_label_item,
            'target_label': tgt_label_item,
            'err_label': self.err_labels[index] if self.err_labels is not None else None
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = max(num_tokens // max(src_len, tgt_len), 1)
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
                'source_label': None,
                'target_label': None
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        # 先根据目标样本排序再根据源样本排序
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        indices = indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

        # 将训练样本按正误分开
        if self.err_labels is not None:
            assert indices.size == self.err_labels.size
            cor_indices = indices[self.err_labels[indices] == 0]
            err_indices = indices[self.err_labels[indices] == 1]
            indices = np.hstack((cor_indices, err_indices))

        return indices
        # return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and getattr(self.tgt, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)
