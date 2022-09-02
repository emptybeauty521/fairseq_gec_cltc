# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('cross_entropy')
class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        copy_alpha = net_output[1]['copy_alpha'].mean().item() if net_output[1]['copy_alpha'] is not None else -1
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'copy_alpha': copy_alpha,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        
        if self.args.positive_label_weight != 1 \
            and sample is not None and sample.get('target_label', None) is not None:
            # return self.compute_weighted_loss(model, net_output, sample, reduce=True)
            loss, _ = self.compute_weighted_loss(model, net_output, sample, reduce=True)
        else:
            lprobs = model.get_normalized_probs(net_output, log_probs=True, sample=sample)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, net_output).view(-1)
            loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                              reduce=reduce)

        # token-level labeling task loss
        src_probs = model.encoder.get_src_probs(net_output, log_probs=True)
        src_label = sample['source_label']
        if src_probs is not None:
            src_probs = src_probs.view(-1, src_probs.size(-1))
            if src_label is not None:
                src_label = src_label.view(-1).long()
            else:
                src_label = torch.zeros(src_probs.size(0)).long()
                if src_probs.device.type == 'cuda':
                    src_label = src_label.cuda()

            # the losses are summed for each minibatch
            src_loss = F.nll_loss(src_probs, src_label, size_average=False, ignore_index=self.padding_idx,
                              reduce=reduce)

            # 忽略dummy/oom/valid batch
            if src_label is None:
                src_loss *= 0

            # loss_val = utils.item(loss.data) if reduce else loss.data
            # src_loss_val = utils.item(src_loss.data) if reduce else src_loss.data

            # loss = loss + ((loss_val / src_loss_val) * self.args.label_gen_loss_rate) * src_loss if src_loss > 0 else src_loss
            loss = loss + self.args.label_gen_loss_rate * src_loss

        return loss, loss

    def compute_weighted_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True, sample=sample)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        # eg: target [1, 3, 5, 7, 9], target_label [0, 1, 0, 1, 0]
        # ==> neg_target [1, -1, 5, -1, 9], pos_target [-1, 3, -1, 7, -1]; 假设padding_idx为-1
        # 计算损失时, ignore_index=self.padding_idx
        target_label = sample['target_label'].view(-1).byte()
        neg_target = target.new_tensor(target).masked_fill_(target_label, self.padding_idx)
        pos_target = target.new_tensor(target).masked_fill_(1-target_label, self.padding_idx)
       
        neg_loss = F.nll_loss(lprobs, neg_target, size_average=False, ignore_index=self.padding_idx,
                              reduce=reduce)
        pos_loss = F.nll_loss(lprobs, pos_target, size_average=False, ignore_index=self.padding_idx,
                              reduce=reduce)

        loss = neg_loss + self.args.positive_label_weight * pos_loss
        # loss = (1/self.args.positive_label_weight) * neg_loss + pos_loss

        return loss, loss


    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        copy_alpha = sum(log.get('copy_alpha', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'copy_alpha': copy_alpha,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
