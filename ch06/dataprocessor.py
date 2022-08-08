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
"""Data preprocessing for transformer."""

import os
import io
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
import nmt


class TrainValDataTransform(object):
    """Transform the machine translation dataset.

    Clip source and the target sentences to the maximum length. For the source sentence, append the
    EOS. For the target sentence, append BOS and EOS.

    Parameters
    ----------
    src_vocab : Vocab
    tgt_vocab : Vocab
    src_max_len : int
    tgt_max_len : int
    """

    def __init__(self, src_vocab, tgt_vocab, src_max_len=None, tgt_max_len=None):
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._src_max_len = src_max_len
        self._tgt_max_len = tgt_max_len

    def __call__(self, src, tgt):
        # For src_max_len < 0, we do not clip the sequence
        if self._src_max_len >= 0:
            src_sentence = self._src_vocab[src.split()[:self._src_max_len]]
        else:
            src_sentence = self._src_vocab[src.split()]
        # For tgt_max_len < 0, we do not clip the sequence
        if self._tgt_max_len >= 0:
            tgt_sentence = self._tgt_vocab[tgt.split()[:self._tgt_max_len]]
        else:
            tgt_sentence = self._tgt_vocab[tgt.split()]
        src_sentence.append(self._src_vocab[self._src_vocab.eos_token])
        tgt_sentence.insert(0, self._tgt_vocab[self._tgt_vocab.bos_token])
        tgt_sentence.append(self._tgt_vocab[self._tgt_vocab.eos_token])
        src_npy = np.array(src_sentence, dtype=np.int32)
        tgt_npy = np.array(tgt_sentence, dtype=np.int32)
        return src_npy, tgt_npy
