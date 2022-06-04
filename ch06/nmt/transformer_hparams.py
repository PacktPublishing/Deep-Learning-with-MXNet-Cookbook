# Transformer Hyper-Parameters
num_units = 512
hidden_size = 2048
dropout = 0.1
epsilon = 0.1
num_layers = 6
num_heads = 8
scaled = True

beam_size = 4
lp_alpha = 0.6
lp_k = 5
bleu = '13a'

# Hyper-parameters for training
optimizer = 'adam'
epochs = 3
batch_size = 2700
test_batch_size = 256
num_accumulated = 1
lr = 2
warmup_steps = 1
save_dir = 'transformer_en_de_u512'
average_start = 1
num_buckets = 20
log_interval = 10
