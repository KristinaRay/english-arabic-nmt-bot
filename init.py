import os
import tqdm
import numpy as np
import requests
import youtokentome as yttm
from zipfile import ZipFile

from config import *
from preprocessing import *
from utils import *

DATA_FILE_PATH = f'{DATA_PATH}/data.zip'
DATA_URL = 'https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/ar-en.txt.zip'
TRG_FILE_NAME = 'OpenSubtitles.ar-en.ar'
SRC_FILE_NAME = 'OpenSubtitles.ar-en.en'
TRG_SAMPLE_FILE_PATH = f'{DATA_PATH}/ar.txt'
SRC_SAMPLE_FILE_PATH = f'{DATA_PATH}/en.txt'
TRG_ORIG_FILE_PATH = f'{DATA_PATH}/{TRG_FILE_NAME}'
SRC_ORIG_FILE_PATH = f'{DATA_PATH}/{SRC_FILE_NAME}'

def fetch_dataset(data_url, data_path, data_file_path):
    
    """ Download data """
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print("Dataset not found, downloading...")
        response = requests.get(data_url, stream=True)
        filename = data_url.split("/")[-1]
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(data_file_path, 'wb') as file:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        log("Download complete")
        log("Extracting...")
        
        zip = ZipFile(DATA_FILE_PATH, "r")
        zip.extract(TRG_FILE_NAME, DATA_PATH)
        zip.extract(SRC_FILE_NAME, DATA_PATH)
        zip.close()
        log("Extracting complete")
        
        num_lines_ar = sum(1 for line in open(TRG_ORIG_FILE_PATH)) # number of lines in arabic file
        num_lines_en = sum(1 for line in open(SRC_ORIG_FILE_PATH)) # number of lines in english file
        
        assert num_lines_ar == num_lines_en, "Lost some data"
        assert os.path.exists(data_path)

    else:

        log('Datasets are found')

def create_sample(sample_size):
    """
    Clean data sample and remove duplicates
    """
    log('Creating txt files for both languages...')
    num_lines_ar = sum(1 for line in open(TRG_ORIG_FILE_PATH)) 
    sample_data_size = 2 * sample_size 
    chosen_lines = set(np.random.choice(np.arange(num_lines_ar), size=sample_data_size, replace=False))
    en_sub = open(SRC_ORIG_FILE_PATH, "r") 
    ar_sub = open(TRG_ORIG_FILE_PATH, "r") 
    unique_pairs = set()
    with open(SRC_TXT_FILE_PATH, "a+") as en, open(TRG_TXT_FILE_PATH, "a+") as ar:
        for idx, (en_line, ar_line) in enumerate(zip(en_sub, ar_sub)):
            if idx in chosen_lines:
                src = clean_en_text(en_line)
                trg = clean_ar_text(ar_line)
                if 2 < len(src) <= MAX_LEN and  2 < len(trg) <= MAX_LEN:
                    if ((src + trg) not in unique_pairs and (len(unique_pairs) < sample_size)): 
                        en.write(src)
                        ar.write(trg)
                        unique_pairs.add((src + trg))
                    elif len(unique_pairs) >= sample_size: 
                        break
    assert len(unique_pairs) == sample_size, "Not enough data"
    en_sub.close()
    ar_sub.close()
    en.close()
    ar.close()
    log("Done")
    log(f'Number of unique pairs of sentences: {len(unique_pairs)}')
        
if __name__ == "__main__":
    
    fetch_dataset(DATA_URL, DATA_PATH, DATA_FILE_PATH)
    create_sample(SAMPLE_SIZE)
    
    log('Training tokenizers...')
    
    yttm.BPE.train(data=TRG_TXT_FILE_PATH, vocab_size=TRG_VOCAB_SIZE, model=TRG_TOKENIZER_PATH)
    yttm.BPE.train(data=SRC_TXT_FILE_PATH, vocab_size=SRC_VOCAB_SIZE, model=SRC_TOKENIZER_PATH)
    
    log("Done")