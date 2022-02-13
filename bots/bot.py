import torch
import youtokentome 
from telegram import Update
from telegram.ext import  Updater, Filters,  MessageHandler, CommandHandler, CallbackContext

from config import *
from models.transformer import Seq2Seq
from utils import translate
from data.preprocessing import *

EN2AR_MODEL_PATH = 'checkpoints_en2ar/checkpoint_epoch_10.pt'
AR2EN_MODEL_PATH = 'checkpoints_ar2en/checkpoint_epoch_10.pt'

TRG_TOKENIZER_PATH = 'data/BPE_arabic_model.bin'
SRC_TOKENIZER_PATH = 'data/BPE_english_model.bin'
en_tokenizer = youtokentome.BPE(SRC_TOKENIZER_PATH)
ar_tokenizer = youtokentome.BPE(TRG_TOKENIZER_PATH)

en2ar_model = Seq2Seq(
    len(en_tokenizer.vocab()),
    len(ar_tokenizer.vocab()), 
    MAX_LEN, PAD_IDX, device)

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
    
en2ar_checkpoint = torch.load(EN2AR_MODEL_PATH, map_location=map_location)
en2ar_model.load_state_dict(en2ar_checkpoint['state_dict']) 
en2ar_model = en2ar_model.to(device)

ar2en_model = Seq2Seq( 
    len(ar_tokenizer.vocab()), 
    len(en_tokenizer.vocab()),
    MAX_LEN, PAD_IDX, device)
    
ar2en_checkpoint = torch.load(AR2EN_MODEL_PATH, map_location=map_location)
ar2en_model.load_state_dict(ar2en_checkpoint['state_dict']) 
ar2en_model = ar2en_model.to(device)

def author_command(update: Update, context: CallbackContext) -> None:
    """Shows the info about the author of the bot."""
    update.message.reply_text("The author of the bot is @kristina_ray")

def start_command(update: Update, context: CallbackContext) -> None:
    """Start the conversation and ask user for input."""
    first_name = update.effective_user.first_name
    update.message.reply_text(
        f"Hi {first_name}! Please, send a sentence in Arabic or English")

def help_command(update: Update, context: CallbackContext) -> None:
    """Displays info on how to use the bot."""
    update.message.reply_text("Use /start to test this bot.")


def handle_text(update: Update, context: CallbackContext) -> None:
    
    text = update.message.text 
    user = update.message.from_user
    
    chat_id = update.message.chat_id
    tokens_to_remove = {PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX}

    if user['id'] != TG_BOT_CHAT_ID:
        msg = f"@{user['username']} {user['id']}"
        context.bot.send_message(TG_BOT_CHAT_ID, msg) 
        context.bot.send_message(TG_BOT_CHAT_ID, text)
        
    
    if clean_ar_text(text) != ' ' and clean_ar_text(text) != '':
        src = ar_tokenizer.encode(text.lower(), bos=True, eos=True)
        trg, attention = translate(ar2en_model, src, device, BOS_IDX, EOS_IDX)
        result = en_tokenizer.decode(trg, ignore_ids=tokens_to_remove)
        context.bot.send_message(chat_id, result[0])
        return None

    elif clean_en_text(text) != ' ' and clean_en_text(text) != '':
        src = en_tokenizer.encode(text.lower(), bos=True, eos=True)
        trg, attention = translate(en2ar_model, src, device, BOS_IDX, EOS_IDX)
        result = ar_tokenizer.decode(trg, ignore_ids=tokens_to_remove)
        context.bot.send_message(chat_id, result[0])
        return None
    else:
        context.bot.send_message(chat_id, "Please, send a text in Arabic or English")
        return None

def main() -> None:
    updater = Updater(TG_BOT_TOKEN, use_context=True)

    # Register the handlers
    disp = updater.dispatcher
    disp.add_handler(CommandHandler('start', start_command))
    disp.add_handler(CommandHandler('help', help_command))
    disp.add_handler(CommandHandler('author', author_command))
    disp.add_handler(MessageHandler(Filters.text, handle_text))
   
    # Start the Bot
    updater.start_polling()
    updater.idle()

if __name__=='__main__':
    main()