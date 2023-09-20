# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Utility functions."""

import os
import inspect
import numpy as np
import math
import mxnet as mx
import gluonnlp as nlp
import time
from tqdm import tqdm_notebook as tqdm

from . import bleu
from . import translation

__all__ = ["evaluate", "translate", "create_vocab"]

# Evaluation function (used also on training loop for validation)
def evaluate(
    data_loader,
    model,
    translator,
    loss_function,
    tgt_vocab,
    ctx):

    translation_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0
    
    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) in enumerate(tqdm(data_loader)):
        
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)
        
        # Calculating Loss
        out, _ = model(
            src_seq,
            tgt_seq[:, :-1],
            src_valid_length,
            tgt_valid_length - 1)

        loss = loss_function(
            out,
            tgt_seq[:, 1:],
            tgt_valid_length - 1).sum().asscalar()
        
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_valid_length - 1).sum().asscalar()
        
        # Translate
        samples, _, sample_valid_length = translator.translate(
            src_seq=src_seq,
            src_valid_length=src_valid_length)
        
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()
        
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                [tgt_vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])
    
    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]
    
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = sentence
    
    return avg_loss, real_translation_out

def translate(
    translator,
    src_seq,
    src_vocab,
    tgt_vocab,
    ctx):
    
    src_sentence = src_vocab[src_seq.split()]
    src_sentence.append(src_vocab[src_vocab.eos_token])
    src_npy = np.array(src_sentence, dtype=np.int32)
    src_nd = mx.nd.array(src_npy)
    src_nd = src_nd.reshape((1, -1)).as_in_context(ctx)
    src_valid_length = mx.nd.array([src_nd.shape[1] - 2]).as_in_context(ctx)

    samples, _, sample_valid_length = translator.translate(
        src_seq=src_nd,
        src_valid_length=src_valid_length)

    max_score_sample = samples[:, 0, :].asnumpy()
    sample_valid_length = sample_valid_length[:, 0].asnumpy()

    translation_out = []
    for i in range(max_score_sample.shape[0]):
        translation_out.append(
            [tgt_vocab.idx_to_token[ele] for ele in
             max_score_sample[i][1:(sample_valid_length[i] - 1)] if ele != 0])
        
    return translation_out

def translate_with_unk(
    translator,
    src_seq,
    src_vocab,
    tgt_vocab,
    ctx):
    
    src_sentence = src_vocab[src_seq.split()]
    src_sentence.append(src_vocab[src_vocab.eos_token])
    src_npy = np.array(src_sentence, dtype=np.int32)
    src_nd = mx.nd.array(src_npy)
    src_nd = src_nd.reshape((1, -1)).as_in_context(ctx)
    src_valid_length = mx.nd.array([src_nd.shape[1] - 2]).as_in_context(ctx)

    samples, _, sample_valid_length = translator.translate(
        src_seq=src_nd,
        src_valid_length=src_valid_length)

    max_score_sample = samples[:, 0, :].asnumpy()
    sample_valid_length = sample_valid_length[:, 0].asnumpy()

    translation_out = []
    for i in range(max_score_sample.shape[0]):
        translation_out.append(
            [tgt_vocab.idx_to_token[ele] for ele in
             max_score_sample[i][1:(sample_valid_length[i] - 1)]])
        
    return translation_out
