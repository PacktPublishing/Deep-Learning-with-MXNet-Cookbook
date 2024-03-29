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
"""Utilities for transformer."""

import numpy as np
import math
import mxnet as mx
import time
import logging
import io
import nmt
import nmt.transformer_hparams as hparams
from tqdm.notebook import tqdm

TOKENS_TO_FILTER = ["<unk>", "<unk>."]

def get_length_index_fn():
    global idx
    idx = 0
    def transform(src, tgt):
        global idx
        result = (src, tgt, len(src), len(tgt), idx)
        idx += 1
        return result
    return transform

def get_data_lengths(dataset):
    return list(dataset.transform(lambda src, tgt: (len(src), len(tgt))))

def compute_loss(model, data_loader, test_loss_function, ctx):
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    """
    avg_loss_denom = 0
    avg_loss = 0.0
    
    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, idx) \
            in enumerate(tqdm(data_loader)):
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)
        
        # Calculating Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        
#         print(src_seq, tgt_seq, src_valid_length, tgt_valid_length)
        
        loss = test_loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1)
        
#         print(loss)
#         print(loss.mean())
#         print(loss.mean().asscalar())
        loss = loss.mean().asscalar()
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)

    # Calculate the average loss and initialize a None-filled translation list
    avg_loss = avg_loss / avg_loss_denom

    # Return the loss
    return avg_loss


def evaluate(model, data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx):
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    translation_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0
    
    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) \
            in enumerate(tqdm(data_loader)):
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)
        
        # Calculating Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = test_loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)
        
#         print(out, src_seq, src_valid_length)
        assert out.context == src_seq.context == src_valid_length.context
        
        # Translate
        samples, _, sample_valid_length = \
            translator.translate(src_seq=src_seq, src_valid_length=src_valid_length)
        
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()

        # Iterate through the tokens and stitch the tokens together for the sentence
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                [tgt_vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])

    # Calculate the average loss and initialize a None-filled translation list
    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]

    # Combine all the words/tokens into a sentence for the final translation
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = detokenizer(nmt.bleu._bpe_to_words(sentence),
                                                return_str=True)

    # Return the loss and the translation
    return avg_loss, real_translation_out

def evaluate_multi(model, data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx_list):
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    assert type(ctx_list) == list
    
    translation_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0
    
    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) \
            in enumerate(tqdm(data_loader)):
        
        src_seq_list = mx.gluon.utils.split_and_load(src_seq, ctx_list=ctx_list, even_split=False)
        tgt_seq_list = mx.gluon.utils.split_and_load(tgt_seq, ctx_list=ctx_list, even_split=False)
        src_valid_length_list = mx.gluon.utils.split_and_load(src_valid_length, ctx_list=ctx_list, even_split=False)
        tgt_valid_length_list = mx.gluon.utils.split_and_load(tgt_valid_length, ctx_list=ctx_list, even_split=False)
        
        samples_list = []
        sample_valid_length_list = []
        
        for src_seq_slice, tgt_seq_slice, src_valid_length_slice, tgt_valid_length_slice in zip(src_seq_list, tgt_seq_list, src_valid_length_list, tgt_valid_length_list):

            # Calculating Loss
            out, _ = model(src_seq_slice, tgt_seq_slice[:, :-1], src_valid_length_slice, tgt_valid_length_slice - 1)
            loss = test_loss_function(out, tgt_seq_slice[:, 1:], tgt_valid_length_slice - 1).mean().asscalar()
            
            avg_loss += loss * (tgt_seq_slice.shape[1] - 1)
            avg_loss_denom += (tgt_seq_slice.shape[1] - 1)
        
            assert out.context == src_seq_slice.context == src_valid_length_slice.context
        
            # Translate
            samples, _, sample_valid_length = \
                translator.translate(src_seq=src_seq_slice, src_valid_length=src_valid_length_slice)
            
            samples_list.append(samples)
            sample_valid_length_list.append(sample_valid_length)
        
        for samples, sample_valid_length in zip(samples_list, sample_valid_length_list):
            max_score_sample = samples[:, 0, :].asnumpy()
            sample_valid_length = sample_valid_length[:, 0].asnumpy()

            # Iterate through the tokens and stitch the tokens together for the sentence
            for i in range(max_score_sample.shape[0]):
                translation_out.append(
                    [tgt_vocab.idx_to_token[ele] for ele in
                     max_score_sample[i][1:(sample_valid_length[i] - 1)]])

        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
    
    # Calculate the average loss and initialize a None-filled translation list
    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]

    # Combine all the words/tokens into a sentence for the final translation
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = detokenizer(nmt.bleu._bpe_to_words(sentence),
                                                return_str=True)

    # Return the loss and the translation
    return avg_loss, real_translation_out

def translate(translator, src_seq, src_vocab, tgt_vocab, detokenizer, ctx):
    src_sentence = src_vocab[src_seq.split()]
    src_sentence.append(src_vocab[src_vocab.eos_token])
    src_npy = np.array(src_sentence, dtype=np.int32)
    src_nd = mx.nd.array(src_npy)
    src_nd = src_nd.reshape((1, -1)).as_in_context(ctx)
    src_valid_length = mx.nd.array([src_nd.shape[1]]).as_in_context(ctx)
    samples, _, sample_valid_length = \
        translator.translate(src_seq=src_nd, src_valid_length=src_valid_length)
    max_score_sample = samples[:, 0, :].asnumpy()
    
    sample_valid_length = sample_valid_length[:, 0].asnumpy()
    translation_out = []
    
    for i in range(max_score_sample.shape[0]):
        translation_out.append(
            [tgt_vocab.idx_to_token[ele] for ele in
             max_score_sample[i][1:(sample_valid_length[i] - 1)]])
    real_translation_out = [None for _ in range(len(translation_out))]
    
    for ind, sentence in enumerate(translation_out):
        translation = detokenizer(
             nmt.bleu._bpe_to_words(sentence),
             return_str=False)
        
        # Filter sentences
        remove_tokens = []
        for word in translation:
            if word in TOKENS_TO_FILTER:
                remove_tokens.append(word)
        
        for token in remove_tokens:
            translation.remove(token)

        real_translation_out[ind] = translation
    
    return real_translation_out              
                
