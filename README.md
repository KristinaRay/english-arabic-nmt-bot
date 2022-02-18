Telegram Bot for English - Arabic Neural Machine Translation

It can be found here t.me/english_arabic_translator_bot

### **Dataset**
The OpenSubtitles dataset for English-Arabic languages is used to train the Seq2Seq model [link to download](https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/ar-en.txt.zip)

### **Data preprocessing**

To download and preprocess a file to remove extra characters and clean up data, run

```
python3 data/get_dataset.py --sample_size 5000000 --max_text_len 150
```
Tokenization is performed using [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe) BPE-tokenizer
### **Model**
Transformer is used as Seq2Seq model

Reference
* Attention Is All You Need [paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

### **Model pruning**

Reference
* https://github.com/lena-voita/the-story-of-heads
* Are Sixteen Heads Really Better than One? [paper](https://blog.ml.cmu.edu/2020/03/20/are-sixteen-heads-really-better-than-one/)

* Analyzing Multi-Head Self-Attention:
Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned
 [paper](https://aclanthology.org/P19-1580.pdf)
 
 * Leaarning Sparse Neural Networks through L0 Regularization
 [paper](https://openreview.net/pdf?id=H1Y8hhg0b)
