import torch

SEED = 42

# Parameters for Transformer & training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 120
HID_DIM = 256
ENC_HEADS = 8
DEC_HEADS = 8
ENC_LAYERS = 3 
DEC_LAYERS = 3 
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

PATIENCE = 5
CLIP = 1
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 5e-5
NUM_WARMUP_STEPS = 30000
SPLIT_RATIO = (0.9, 0.05)

# Parameters for youtokentome BPE tokenizer

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
SRC_VOCAB_SIZE = 32000
TRG_VOCAB_SIZE = 24000

# Paths or parameters for data

SAMPLE_SIZE = 3000000

DATA_PATH = 'data'
LOG_FILE_PATH = 'logs'
CHECKPOINT_PATH = 'checkpoints'
TRG_TXT_FILE_PATH = f'{DATA_PATH}/ar.txt'
SRC_TXT_FILE_PATH = f'{DATA_PATH}/en.txt'
TRG_TOKENIZER_PATH = f'{DATA_PATH}/BPE_arabic_model.bin'
SRC_TOKENIZER_PATH = f'{DATA_PATH}/BPE_english_model.bin'

MODEL_PATH = f'{CHECKPOINT_PATH}/best_loss_model.pt'

# Parameters for telegram notification and translation bots

TG_NOTIFY_CHAT_ID = 0
TG_NOTIFY_TOKEN = ''
TG_BOT_TOKEN = ''
TG_BOT_CHAT_ID = 0