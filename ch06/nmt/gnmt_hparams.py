# GNMT Hyper-Parameters
num_hidden = 512
num_layers = 2
num_bi_layers = 1
dropout = 0.2

beam_size = 10
lp_alpha = 1.0
lp_k = 5
bleu = '13a'

# Hyper-parameters for training
batch_size, test_batch_size = 64, 32
num_buckets = 5
epochs = 8
clip = 5
lr = 0.001
lr_update_factor = 0.5
log_interval = 100
save_dir = "gnmt_en_de_512"
