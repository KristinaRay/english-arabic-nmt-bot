import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from transformers import get_cosine_schedule_with_warmup
import time
import matplotlib
import math
import numpy as np
import youtokentome 
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from models.transformer import Seq2Seq
from data.dataloader import *
from config import *
from utils import *

trg_tokenizer = youtokentome.BPE(TRG_TOKENIZER_PATH)
src_tokenizer = youtokentome.BPE(SRC_TOKENIZER_PATH)
src_data = []
trg_data = []
with open(SRC_TXT_FILE_PATH, "r") as source:
    for src_line in source:
        src_data.append(src_tokenizer.encode(src_line[:-1]))
with open(TRG_TXT_FILE_PATH, "r") as target:
    for trg_line in target:
        trg_data.append(trg_tokenizer.encode(trg_line[:-1]))

log(f"Number of tokens in source language: {len(trg_tokenizer.vocab())}")
log(f"Number of tokens in target language: {len(src_tokenizer.vocab())}")

token_frequency_distribution(src_tokenizer, SRC_TXT_FILE_PATH, LOG_FILE_PATH, "English")
token_frequency_distribution(trg_tokenizer, TRG_TXT_FILE_PATH, LOG_FILE_PATH, "Arabic")

dataset = list(zip(src_data, trg_data))
train_size, val_size = int(SPLIT_RATIO[0] * len(dataset)), int(SPLIT_RATIO[1] * len(dataset))
train_data, valid_data, test_data = random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size])

log(f"Number of training examples: {len(train_data)}")
log(f"Number of validation examples: {len(valid_data)}")
log(f"Number of testing examples: {len(test_data)}")

train_loader = Dataloader(train_data, batch_size=128, shuffle=True)
valid_loader = Dataloader(valid_data, batch_size=128, shuffle=False) 
test_loader = Dataloader(test_data, batch_size=128, shuffle=False)
log(f"Number of training batches: {len(train_loader)}")
log(f"Number of validation batches: {len(valid_loader)}")
log(f"Number of testing batches: {len(test_loader)}")

model = Seq2Seq(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, MAX_LEN, PAD_IDX, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=len(train_loader) * NUM_EPOCHS, num_cycles=1)

log('Training the model...')
log(f'The model has {count_parameters(model):,} trainable parameters')
start_time = time.time()
epoch=EPOCH_NUM
log(f'Training epoch: {epoch+1:02}')

loss_history, bleu_history, train_loss, train_bleu_score = train(model, train_loader, optimizer, scheduler, criterion, src_tokenizer, trg_tokenizer, clip=CLIP, device=device, pruning=False)
valid_loss_history, valid_bleu_history, valid_loss, valid_bleu_score = evaluate(model, valid_loader, criterion, src_tokenizer, trg_tokenizer, device=device, pruning=False)
end_time = time.time()
epoch_mins, epoch_secs = epoch_time(start_time, end_time)
checkpoint = {
    'epoch': epoch + 1,
    'train_loss': train_loss,
    'train_bleu_score': train_bleu_score,
    'valid_loss': valid_loss,
    'valid_bleu_score': valid_bleu_score,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
}
np.save(f'{LOG_FILE_PATH}/loss_history_epoch_{epoch}', loss_history)
np.save(f'{LOG_FILE_PATH}/bleu_history_epoch_{epoch}', bleu_history)
np.save(f'{LOG_FILE_PATH}/valid_loss_history_epoch_{epoch}', valid_loss_history)
np.save(f'{LOG_FILE_PATH}/valid_bleu_history_epoch_{epoch}', valid_bleu_history)
save_checkpoints(checkpoint, epoch, best_loss=False, best_bleu_score=False, checkpoint_path='checkpoints_en2ar')
log(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n'
    f'Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Train Bleu Score: {train_bleu_score:1.3f}\n'  
    f'Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} | Val. Bleu Score: {valid_bleu_score:1.3f}\n') 