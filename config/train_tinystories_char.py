# train a mini gpt model on the tinystories dataset

out_dir = 'out-tinystories-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'poormans-nanogpt'
wandb_run_name = 'mini-gpt'

dataset = 'tinystories'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 512 # context of up to 256 previous characters

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 15000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

device = 'cuda' 
compile = True 
