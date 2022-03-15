**Telegram Bot for English - Arabic Neural Machine Translation**

It can be found here -> [https://t.me/english_arabic_translator_bot](https://t.me/english_arabic_translator_bot)

The bot is deployed on [Oracle Cloud](https://www.oracle.com/index.html)

### **Dataset**
The OpenSubtitles dataset for English-Arabic languages is used to train the Seq2Seq model [link to download](https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/ar-en.txt.zip)

### **Data preprocessing**

To download and preprocess a file in order to remove extra characters and clean up data, run

```
python data/get_dataset.py --sample_size 5000000 --max_text_len 150
```
Tokenization is performed using [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe) BPE-tokenizer
### **Model**
The implementation of the Transformer in PyTorch with 6 layered decoder and encoder and 8 multi attention heads with Glorot initialized parameters. 

Reference
* Attention Is All You Need [paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
* Understanding the difficulty of training deep feedforward neural networks [paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

### **Learning rate**
For the training learning rate 0.00005 is used with warm up for 30000 iterations
![alt text](https://github.com/KristinaRay/english-arabic-nmt-bot/blob/main/pics/learning_rate.png)

### **Model pruning**
The implementation of the method Voita et al. in PyTorch [paper](https://aclanthology.org/P19-1580.pdf)

2 experiments of the model attention heads pruning were carried out
with λ = 0.05 [experiment_1](https://github.com/KristinaRay/english-arabic-nmt-bot/tree/main/experiment_1) and λ = 0.01 [experiment_2](https://github.com/KristinaRay/english-arabic-nmt-bot/tree/main/experiment_2)

### **Results**

For λ = 0.05 91 retained heads, for λ = 0.01 89 retained heads.

![alt text](https://github.com/KristinaRay/english-arabic-nmt-bot/blob/main/experiment_1/assets/enc_self_attn_gates.gif)
![alt text](https://github.com/KristinaRay/english-arabic-nmt-bot/blob/main/experiment_2/assets/enc_self_attn_gates.gif)

Reference
* https://github.com/lena-voita/the-story-of-heads
* Are Sixteen Heads Really Better than One? [paper](https://blog.ml.cmu.edu/2020/03/20/are-sixteen-heads-really-better-than-one/)
* Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned [paper](https://aclanthology.org/P19-1580.pdf)
* Learning Sparse Neural Networks through L0 Regularization [paper](https://openreview.net/pdf?id=H1Y8hhg0b)
