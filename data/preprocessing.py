import re

def replace_contractions(sentence: str) -> str:
    """
    reference
    https://www.enchantedlearning.com/grammar/contractions/list.shtml
    https://www.analyticsvidhya.com/blog/2021/06/must-known-techniques-for-text-preprocessing-in-nlp/
    """
    contractions_dict = {
                      "'cause": "because",
                      "'em": "them",
                      "'tis": "it is",
                      "'twas": "it was",
                      "I'd": "I would",
                      "I'd've": "I would have",
                      "I'll": "I will",
                      "I'll've": "I will have",
                      "I'm": "I am",
                      "I'm'a": "I am about to",
                      "I'm'o": "I am going to",
                      "I've": "I have",
                      "Whatcha": "What are you",
                      "ain't": "are not",
                      "amn't": "am not",
                      "aren't": "are not",
                      "can't": "cannot",
                      "could've": "could have",
                      "couldn't": "could not",
                      "daren't": "dare not",
                      "daresn't": "dare not",
                      "dasn't": "dare not",
                      "didn't": "did not",
                      "doesn't": "does not",
                      "don't": "do not",
                      "everyone's": "everyone is",
                      "finna": "fixing to",
                      "gimme": "give me",
                      "gon't": "go not",
                      "gonna": "going to",
                      "gotta": "got to",
                      "hadn't": "had not",
                      "hasn't": "has not",
                      "haven't": "have not",
                      "he'd": "he had",
                      "he'll": "he will",
                      "he's": "he is",
                      "here's": "here is",
                      "how'd": "how did",
                      "how'd'y": "how do you",
                      "how'll": "how will",
                      "how're": "how are",
                      "how's": "how is",
                      "i'd": "i would",
                      "i'd've": "i would have",
                      "i'll": "i will",
                      "i'll've": "i will have",
                      "i'm": "i am",
                      "i've": "i have",
                      "isn't": "is not",
                      "it'd": "it would",
                      "it'd've": "it would have",
                      "it'll": "it will",
                      "it'll've": "it will have",
                      "it's": "it is",
                      "kinda": "kind of",
                      "let's": "let us",
                      "luv": "love",
                      "ma'am": "madam",
                      "may've": "may have",
                      "mayn't": "may not",
                      "might've": "might have",
                      "mightn't": "might not",
                      "mightn't've": "might not have",
                      "must've": "must have",
                      "mustn't": "must not",
                      "mustn't've": "must not have",
                      "ne'er": "never",
                      "needn't": "need not",
                      "needn't've": "need not have",
                      "o'": "of",
                      "o'clock": "of the clock",
                      "ol'": "old",
                      "oughtn't": "ought not",
                      "oughtn't've": "ought not have",
                      "sha'n't": "shall not",
                      "shan't": "shall not",
                      "shan't've": "shall not have",
                      "she'd": "she had",
                      "she'd've": "she would have",
                      "she'll": "she will",
                      "she'll've": "she will have",
                      "she's": "she is",
                      "should've": "should have",
                      "shouldn't": "should not",
                      "shouldn't've": "should not have",
                      "so's": "so is",
                      "so've": "so have",
                      "somebody's": "somebody is",
                      "someone's": "someone is",
                      "something's": "something is",
                      "that'd've": "that would have",
                      "that'll": "that will",
                      "that's": "that is",
                      "there'd": "there would",
                      "there'd've": "there would have",
                      "there'll": "there will",
                      "there're": "there are",
                      "there's": "there is",
                      "these're": "these are",
                      "they'd": "they would",
                      "they'd've": "they would have",
                      "they'll": "they will",
                      "they'll've": "they will have",
                      "they're": "they are",
                      "they've": "they have",
                      "those're": "those are",
                      "to've": "to have",
                      "wanna": "want to",
                      "wasn't": "was not",
                      "we'd": "we would",
                      "we'd've": "we would have",
                      "we'll": "we will",
                      "we'll've": "we will have",
                      "we're": "we are",
                      "we've": "we have",
                      "weren't": "were not",
                      "what'd": "what would",
                      "what'll": "what will",
                      "what'll've": "what will have",
                      "what're": "what are",
                      "what's": "what is",
                      "what've": "what have",
                      "when'd": "when would",
                      "when'll": "when will",
                      "when's": "when is",
                      "when've": "when have",
                      "where'd": "where would",
                      "where'll": "where will",
                      "where's": "where is",
                      "which's": "which is",
                      "who'd": "who would",
                      "who'd've": "who would have",
                      "who'll": "who will",
                      "who'll've": "who will have",
                      "who're": "who are",
                      "who's": "who is",
                      "why'd": "why would",
                      "why'll": "why will",
                      "why're": "why are",
                      "why's": "why is",
                      "why've": "why have",
                      "will've": "will have",
                      "won't": "will not",
                      "won't've": "will not have",
                      "would've": "would have",
                      "wouldn't": "would not",
                      "wouldn't've": "would not have",
                      "y'all": "you all",
                      "y'all'd": "you all would",
                      "y'all'd've": "you all would have",
                      "y'all're": "you all are",
                      "y'all've": "you all have",
                      "you'd": "you would",
                      "you'd've": "you would have",
                      "you'll": "you will",
                      "you'll've": "you will have",
                      "you're": "you are",
                      "you've": "you have"                
    }
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    def replace(match):
        return contractions_dict[match.group(0)] 
    return contractions_re.sub(replace, sentence)
    
def replace_misspell(sentence: str) -> str:
    """
    reference
    https://www.macmillandictionary.com/misspells.html
    """
    mispell_dict = {
                  'colour': 'color',
                  'centre': 'center',
                  'favourite': 'favorite', 
                  'travelling': 'traveling', 
                  'counselling': 'counseling',
                  'theatre': 'theater', 
                  'cancelled': 'canceled', 
                  'labour': 'labor', 
                  'organisation': 'organization', 
                  'wwii': 'world war 2',
                  'citicise': 'criticize', 
                  'youtu ': 'youtube ', 
                  'Qoura': 'Quora', 
                  'sallary': 'salary', 
                  'Whta': 'What', 
                  'narcisist': 'narcissist', 
                  'howdo': 'how do', 
                  'whatare': 'what are', 
                  'howcan': 'how can', 
                  'howmuch': 'how much', 
                  'howmany': 'how many', 
                  'whydo': 'why do', 
                  'doI': 'do I', 
                  'theBest': 'the best', 
                  'howdoes': 'how does', 
                  'Etherium': 'Ethereum', 
                  'narcissit': 'narcissist', 
                  'bigdata': 'big data', 
                  '2k17': '2017', 
                  '2k18': '2018', 
                  'qouta': 'quota', 
                  'exboyfriend': 'ex boyfriend', 
                  'airhostess': 'air hostess',
                  'whst': 'what', 
                  'watsapp': 'whatsapp',
                  'demonitisation': 'demonetization',
                  'demonitization': 'demonetization',
                  'demonetisation': 'demonetization',
                  'accomodation': 'accommodation',
                  'adress': 'address',
                  'accomodate': 'accommodate',
                  'wether': 'weather', # whether 
                  'rehersal': 'rehearsal',
                  'commited': 'committed',
                  'persue': 'pursue',
                  'occurence': 'occurrence',
                  'lenght': 'length',
                  'strenght': 'strength',
                  'seperate': 'separate',
                  'appaling': 'appalling',
                  'tought': 'thought', # taught
                  'throught': 'through',
                  'commision': 'commission',
                  'comission': 'commission',
                  'recieve': 'receive', 
                  'collegue': 'colleague',
                  'desease': 'disease',
                  'compell': 'compel',
                  'bizzare': 'bizarre',
                  'concious': 'conscious',
                  'advertisment': 'advertisement',
                  'succint': 'succinct',
                  'rythm': 'rhythm',
                  'wich': 'which', # witch
                  'wheather': 'weather', # whether 
                  'percieve': 'perceive',
                  'occure': 'occur',
                  'enterpreneur': 'entrepreneur',
                  'aquire': 'acquire',
                  'convinient': 'convenient',
                  'devide': 'divide',
                  'agressive': 'aggressive',
                  'enviroment': 'environment',
                  'supress': 'suppress',
                  'embarassed': 'embarrassed',
                  'miniscule':'minuscule',
                  'occured': 'occurred',
                  'strech': 'stretch',
                  'embarrased': 'embarrassed',
                  'responsability': 'responsibility',
                  'assesment': 'assessment',
                  'akward': 'awkward',
                  'endevour': 'endeavour',
                  'belive': 'believe',
                  'wierd ': 'weird',
                  'achive': 'achieve',
                  'greatful': 'grateful',
                  'biogrophay':'biography'
    }

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    def replace(match):
        return mispell_dict[match.group(0)] 
    return mispell_re.sub(replace, sentence)

def replace_endings(sentence: str) -> str:
    """
    reference
    https://github.com/kootenpv/contractions/blob/master/contractions/data/leftovers_dict.json
    """
    endings_dict = {
                  "'all": " all",
                  "'am": " am",
                  "'cause": "because",
                  "'d": " would",
                  "'ll": " will",
                  "'re": " are",
                  "'em": "them",
                  "'er": " her",
                  "doin'": "doing",
                  "goin'": "going",
                  "nothin'": "nothing",
                  "somethin'": "something",
                  "havin'": "having",
                  "lovin'": "loving",
                  "'coz": "because",
                  "thats": "that is",
                  "whats": "what is"
    }
    endings_re = re.compile('(%s)' % '|'.join(endings_dict.keys()))
    def replace(match):
        return endings_dict[match.group(0)] 
    return endings_re.sub(replace, sentence)

def replace_slang(sentence: str) -> str:
    """
    reference
    https://github.com/kootenpv/contractions/blob/master/contractions/data/slang_dict.json
    """
    slang_dict = {
                "'aight": "alright",
                "dunno": "do not know",
                "howdy": "how do you do",
                "ima": "I am going to",
                "innit": "is it not",
                "iunno": "I do not know",
                "g'day": "good day",
                "gonna": "going to",
                "gotta": "got to",
                "wanna": "want to",
                "woulda": "would have",
                "gimme": "give me",
                "asap": "as soon as possible",
                " u ": " you ",
                " r ": " are "
    }
    slang_re = re.compile('(%s)' % '|'.join(slang_dict.keys()))
    def replace(match):
        return slang_dict[match.group(0)] 
    return slang_re.sub(replace, sentence)


def preprocess_sentence(sentence: str) -> str:
    '''
    Lowercase, trim, and remove non-letter and non-digit characters
    ''' 
    sentence = sentence.lower()
    clean = lambda x: x in  "abcdefghijklmnopqrstuvwxyz '\n"
    sentence = ''.join([i for i in  sentence if  clean(i)])
    sentence = re.sub(' +', ' ', sentence)
    return sentence

def clean_en_text(sentence: str) -> str:
    preprocess = [replace_contractions, replace_misspell, replace_endings, replace_slang, preprocess_sentence]
    for f in preprocess:
        sentence = f(sentence)
    return sentence
    
def clean_ar_text(sentence: str) -> str:
    """
    reference
    https://jrgraphix.net/r/Unicode/0600-06FF
    https://en.wikipedia.org/wiki/Arabic_alphabet
    """
    # all_diacritics = u"[\u0640\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670]"
    # remove_diacritics = lambda x: re.sub(all_diacritics, '', x)
    sentence = re.sub(f"[^{'؀-ۿ'} ,\n]", '', sentence)
    sentence = re.sub(f"['.,؟?!،]", '', sentence)
    sentence = re.sub(' +', ' ', sentence)
    return sentence