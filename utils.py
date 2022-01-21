import torch
from torch import nn
import gc
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from notify import TG
from config import *


tg = TG(TG_NOTIFY_TOKEN, TG_NOTIFY_CHAT_ID)

def seed_everything(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def token_frequency_distribution(tokenizer, data_path, output_data_path, language):
    
    data = [line for line in open(data_path)]
    tokens = [tokenizer.encode(line) for line in data]
    token_counts = np.bincount([token_id for token in tokens for token_id in token])
    os.makedirs(output_data_path, exist_ok=True)
    plt.hist(token_counts, bins=100)
    plt.title(f'{language} Token Frequency Distribution')
    plt.yscale('log');
    plt.savefig(os.path.join(output_data_path, f'{language}_token_frequency_distribution.png'))
    tg.send_photo(os.path.join(output_data_path, f'{language}_token_frequency_distribution.png'))
    plt.close()
    
def count_parameters(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        
def save_checkpoints(checkpoint, epoch_number, best_loss, best_bleu_score, checkpoint_path):
    
    os.makedirs(checkpoint_path, exist_ok=True)
    if best_loss:
        path = os.path.join(checkpoint_path, 'best_loss_model.pt')
        torch.save(checkpoint, path)
    elif best_bleu_score:
        path = os.path.join(checkpoint_path, 'best_bleu_score_model.pt')
        torch.save(checkpoint, path)
    else:
        path = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch_number}.pt')
        torch.save(checkpoint, path)

def calculate_bleu(references, hypotheses):
    """
    The default BLEU calculates a score for up to 4-grams
    (this is called BLEU-4).
    reference:
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    """
    list_of_references = [[text.split()] for text in references]
    hypotheses = [text.split() for  text in hypotheses]
    smooth = SmoothingFunction()
    return corpus_bleu(list_of_references, hypotheses, smoothing_function=smooth.method1, weights = (0.5, 0.5)) 

def log(text):
    
    os.makedirs(LOG_FILE_PATH, exist_ok=True)
    tg.send_message(text)
    with open(os.path.join(LOG_FILE_PATH,'log.txt'), 'a+') as file: 
        file.write(f'{text}\n')
    
def epoch_time(start_time, end_time):
    
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def plotting(history, train_history, valid_history, train_history_perplexity, valid_history_perplexity, train_bleu_scores, valid_bleu_scores, output_data_path): 
    
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 6))

    clear_output(True)
    ax[0].plot(history, label='train loss')
    ax[0].set_xlabel('Batch')
    ax[0].set_title('Train loss')
    if train_history is not None:
        ax[1].plot(train_history, label='general train history')
        ax[1].set_xlabel('Epochs')
        ax[1].set_title('Loss')
        ax[1].legend()
    if valid_history is not None:
        ax[1].plot(valid_history, label='general valid history')
        ax[1].legend()
    ax[2].plot(train_history_perplexity, label='train perplexity')
    ax[2].set_xlabel('Epochs')
    ax[2].set_title('Perplexity')
    ax[2].plot(valid_history_perplexity, label='valid perplexity')
    ax[3].plot(train_bleu_scores, label='train blue score')
    ax[3].set_xlabel('Epochs')
    ax[3].set_title('BLEU Score')
    ax[3].plot(valid_bleu_scores, label='valid blue score')

    plt.legend()
    plt.savefig(os.path.join(output_data_path, 'training_history.png'))
    tg.send_photo(os.path.join(output_data_path, 'training_history.png'))
    plt.close()
    
def train(model, iterator, optimizer, scheduler, criterion, src_tokenizer, trg_tokenizer, clip, device):
    
    model.train()
    epoch_loss = 0
    epoch_bleu_score = 0
    history = []
    for i, batch in enumerate(iterator):
        
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output_ = output.contiguous().view(-1, output_dim)
        trgs = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
        
        indexes = {PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX}
        source = src_tokenizer.decode(src.tolist(), ignore_ids=indexes)
        targets = trg_tokenizer.decode(trg.tolist(), ignore_ids=indexes)
        outputs = trg_tokenizer.decode(output.argmax(-1).tolist(), ignore_ids=indexes)
        batch_bleu = calculate_bleu(targets, outputs)
        
        if i % 1000 == 0:
            idx = np.random.choice(np.arange(len(outputs)))
            log(f"English: {source[idx]}")
            log(f"Original: {targets[idx]}")
            log(f"Generated: {outputs[idx]}")
            
        loss = criterion(output_, trgs)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
        epoch_loss += loss.item()
        epoch_bleu_score += batch_bleu
        
        history.append(loss.cpu().data.numpy())
        
        del batch
        gc.collect()        
    return history, epoch_loss / len(iterator), epoch_bleu_score / len(iterator)

def evaluate(model, iterator, criterion, src_tokenizer, trg_tokenizer, device):
    
    model.eval()
        
    epoch_loss = 0
    epoch_bleu_score = 0
    with torch.no_grad():
        
        for i, batch in enumerate(iterator):
    
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
    
            output, _ = model(src, trg[:,:-1])
                
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
                
            output_dim = output.shape[-1]
                
            output_ = output.contiguous().view(-1, output_dim)
            trgs = trg[:,1:].contiguous().view(-1)
                
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            indexes = {PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX}
            source = src_tokenizer.decode(src.tolist(), ignore_ids=indexes)
            targets = trg_tokenizer.decode(trg.tolist(), ignore_ids=indexes)
            outputs = trg_tokenizer.decode(output.argmax(-1).tolist(), ignore_ids=indexes)
            batch_bleu = calculate_bleu(targets, outputs)
            
            loss = criterion(output_, trgs)
            epoch_loss += loss.item()
            epoch_bleu_score += batch_bleu
            
            del batch
            gc.collect()
            
    return epoch_loss / len(iterator),  epoch_bleu_score / len(iterator)
    
def translate(model, sentence, device, bos_idx, eos_idx, max_len=100):
        
    model.eval()
    src_tensor = torch.LongTensor(sentence).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    trg = [bos_idx]
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(-1)[:,-1].item()
        trg.append(pred_token)

        if pred_token == eos_idx:
            break
    
    return trg, attention
    
    
def load_checkpoint(model, checkpoint_path, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
      