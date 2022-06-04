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
import time
from tqdm import tqdm

from . import utils
from . import bleu

__all__ = ["get_length_index_fn", "evaluate", "translate",
           "train_one_epoch"]


def get_length_index_fn():
    global idx
    idx = 0
    def transform(src, tgt):
        global idx
        result = (src, tgt, len(src), len(tgt), idx)
        idx += 1
        return result
    return transform

def evaluate(model, data_loader, test_loss_function, translator, tgt_vocab, detokenizer, context):
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
        
        # Data on GPU
        src_seq = src_seq.as_in_context(context)
        tgt_seq = tgt_seq.as_in_context(context)
        src_valid_length = src_valid_length.as_in_context(context)
        tgt_valid_length = tgt_valid_length.as_in_context(context)
        
        # Calculating Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = test_loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)
        
        # Translate
        samples, _, sample_valid_length = \
            translator.translate(src_seq=src_seq, src_valid_length=src_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()
        
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                [tgt_vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])
    
    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]
    
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = detokenizer(bleu._bpe_to_words(sentence),
                                                return_str=True)
    
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
        real_translation_out[ind] = detokenizer(bleu._bpe_to_words(sentence),
                                                return_str=True)
    return real_translation_out              

def train(model, train_data_loader, valid_data_loader, loss_function, trainer, translator,
          tgt_vocab, detokenizer, save_dir, hparams, ctx):
    
    best_valid_bleu = 0.0
    train_data_loader_length = len(train_data_loader)

    # Run through each epoch
    for epoch_id in tqdm(range(hparams.epochs)):

        log_avg_loss = 0
        log_avg_gnorm = 0
        log_wc = 0
        log_start_time = time.time()

        # Iterate through each batch
        for batch_id, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, _) \
                in enumerate(tqdm(train_data_loader)):

            src_seq = src_seq.as_in_context(ctx)
            tgt_seq = tgt_seq.as_in_context(ctx)
            src_valid_length = src_valid_length.as_in_context(ctx)
            tgt_valid_length = tgt_valid_length.as_in_context(ctx)

            # Compute gradients and losses
            with mx.autograd.record():
                out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
                loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean()
                loss = loss * (tgt_seq.shape[1] - 1) / (tgt_valid_length - 1).mean()
                loss.backward()

            grads = [p.grad(ctx) for p in model.collect_params().values()]
            gnorm = mx.gluon.utils.clip_global_norm(grads, hparams.clip)
            trainer.step(1)
            src_wc = src_valid_length.sum().asscalar()
            tgt_wc = (tgt_valid_length - 1).sum().asscalar()
            step_loss = loss.asscalar()
            log_avg_loss += step_loss
            log_avg_gnorm += gnorm
            log_wc += src_wc + tgt_wc
            if (batch_id + 1) % hparams.log_interval == 0:
                wps = log_wc / (time.time() - log_start_time)
                print("[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, gnorm={:.4f}, "
                             "throughput={:.2f}K wps, wc={:.2f}K"
                             .format(epoch_id, batch_id + 1, train_data_loader_length,
                                     log_avg_loss / hparams.log_interval,
                                     np.exp(log_avg_loss / hparams.log_interval),
                                     log_avg_gnorm / hparams.log_interval,
                                     wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_avg_gnorm = 0
                log_wc = 0

        # Evaluate the losses on validation and test datasets and find the corresponding BLEU score and log it
        valid_loss, valid_translation_out = utils.evaluate(
            model,
            valid_data_loader,
            loss_function,
            translator,
            tgt_vocab,
            detokenizer,
            ctx)
        
        valid_bleu_score, _, _, _, _ = bleu.compute_bleu([val_tgt_sentences], valid_translation_out)
        
        print("[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}"
                     .format(epoch_id, valid_loss, np.exp(valid_loss), valid_bleu_score * 100))

        # Save the model if the BLEU score is better than the previous best
        if valid_bleu_score > best_valid_bleu:
            best_valid_bleu = valid_bleu_score
            save_path = os.path.join(save_dir, "valid_best.params")
            print("Save best parameters to {}".format(save_path))
            model.save_parameters(save_path)

        # Update the learning rate based on the number of epochs that have passed
        if epoch_id + 1 >= (epochs * 2) // 3:
            new_lr = trainer.learning_rate * hparams.lr_update_factor
            print("Learning rate change to {}".format(new_lr))
            trainer.set_learning_rate(new_lr)
