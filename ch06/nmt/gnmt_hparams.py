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

# GNMT Hyper-Parameters
num_hidden = 512
num_layers = 2
num_bi_layers = 1
dropout = 0.2

beam_size = 10
lp_alpha = 1.0
lp_k = 5
bleu = "tweaked"
# BeamSearchSampler Maximum Length Search
max_length = 150

# Hyper-parameters for training
optimizer = "adam"
batch_size, test_batch_size = 128, 64
num_buckets = 5
epochs = 12
clip = 5
lr = 0.001
lr_update_factor = 0.5
log_interval = 100
file_name = "gnmt_vi_en_512.params"
