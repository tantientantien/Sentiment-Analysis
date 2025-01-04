import re
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from contractions import contractions_dict
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.word_to_idx = {}
    
    def expand_contractions(self, text):
        pattern = re.compile('|'.join(contractions_dict.keys()))
        return pattern.sub(lambda x: contractions_dict[x.group()], text)
    
    def load_data(self, filepath):
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
        df = df.dropna(subset=['text'])
        return df['text'].tolist(), df['sentiment'].tolist()
    
    def preprocess(self, sentence):
        # lowercase
        sentence = sentence.lower()
        
        # expand contractions
        sentence = self.expand_contractions(sentence)
        
        # remove URLs
        sentence = re.sub(r'http\S+|www\S+|https\S+', '', sentence, flags=re.MULTILINE)
        
        # remove punctuation except emojis (if keep emojis)
        sentence = sentence.translate(str.maketrans('', '', string.punctuation.replace('!', '').replace('?', '')))
        
        # tokenize
        tokens = nltk.word_tokenize(sentence)
        
        # remove stopwords
        important_words = {'not', 'never', 'no', 'none', 'nobody', 'nothing', 
                        'nowhere', 'neither', 'nor', 'very', 'extremely', 
                        'absolutely', 'always', 'too', 'barely', 'hardly', 
                        'scarcely', 'little', 'few', 'hate', 'love', 
                        'dislike', 'prefer'}

        tokens = [word for word in tokens if word not in self.stop_words or word in important_words]
        
        # lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return tokens
    
    def build_vocab(self, sentences, min_freq=1):
        word_freq = {}
        for sentence in sentences:
            for word in sentence:
                word_freq[word] = word_freq.get(word, 0) + 1
        self.word_to_idx = {word: idx for idx, (word, freq) in enumerate(word_freq.items()) if freq >= min_freq}
        self.word_to_idx['<PAD>'] = len(self.word_to_idx)
        self.word_to_idx['<UNK>'] = len(self.word_to_idx)
    
    def save_word_to_idx(self, filepath='models/word_to_idx.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self.word_to_idx, f)

def main():
    preprocessor = TextPreprocessor()
    texts, _ = preprocessor.load_data('dataset/train.csv')  
    tokenized_sentences = [preprocessor.preprocess(text) for text in texts]  
    preprocessor.build_vocab(tokenized_sentences, min_freq=1)
    preprocessor.save_word_to_idx()

if __name__ == "__main__":
    main()
    