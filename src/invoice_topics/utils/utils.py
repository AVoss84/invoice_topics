import nltk, re, string, gensim, spacy        # for type hints
from string import punctuation
from nltk.corpus import stopwords
#from nltk.cluster.util import cosine_distance
from nltk.stem.snowball import SnowballStemmer
#from pydantic import BaseModel
from nltk.tokenize import word_tokenize, RegexpTokenizer
import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from typing import (Dict, List, Text, Optional, Any, Callable, Union)
from gensim.models import FastText, Phrases, phrases, TfidfModel
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
from gensim import corpora
from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
from gensim.parsing.preprocessing import stem_text, strip_multiple_whitespaces, strip_short, strip_non_alphanum, strip_punctuation, strip_numeric
from copy import deepcopy
#from tqdm.auto import tqdm
from tqdm import tqdm
import pandas as pd
from src.services import file
from src.config import global_config as glob

nltk.download('punkt')
nltk.download('stopwords')


class clean_text(BaseEstimator, TransformerMixin):

    def __init__(self, verbose : bool = True, language : str = 'german', stem : bool = False, lemma : bool = False, **kwargs):
        self.verbose = verbose
        self.kwargs = kwargs
        self.stem = stem
        self.lemma = lemma
        self.stop_words = set(stopwords.words(language))
        if self.verbose: print(f'Using {len(self.stop_words)} stop words.')
        try:
            self.german_stopwords = file.JSONservice(verbose=False, root_path = glob.UC_CODE_DIR, path = 'config/stopwords.json').doRead()
            self.stop_words = self._add_stopwords(self.german_stopwords)
        except Exception as ex:
            print(ex); self.german_stopwords = []
        if self.verbose: print(f'Adding custom German stop words...') 

        if 'without_stopwords' in list(self.kwargs.keys()):
            self.stop_words = self._remove_stopwords(self.kwargs.get('without_stopwords', ''))
                
        if 'with_stopwords' in list(self.kwargs.keys()):
            self.stop_words = self._add_stopwords(self.kwargs.get('with_stopwords', '')) 
            
        if self.stem:
            self.stemmer = SnowballStemmer(language); print("Loading nltk stemmer.")
            
        if self.lemma:
            self.nlp = spacy.load('de_core_news_lg'); print("Loading spaCy embeddings for lemmatization.")
            
        self.umlaut = file.YAMLservice(root_path = glob.UC_CODE_DIR, path = 'config/preproc_txt.yaml').doRead()

    def _add_stopwords(self, new_stopwords : Union[List, None])-> set:
        """
        Change stopword list. Include into stopword list

        Args:
            new_stopwords (list): _description_

        Returns:
            set: _description_
        """
        if new_stopwords:
            old = self.stop_words.copy()
            self.stop_words = self.stop_words.union(set(new_stopwords))
            if self.verbose: print(f"Added {len(self.stop_words)-len(old)} stopword(s).")
            return self.stop_words

    def _remove_stopwords(self, without_stopwords : Union[List, None])-> set:
        """
        Change stopword list. Exclude from stopwords

        Args:
            without_stopwords (list): _description_

        Returns:
            set: _description_
        """
        if without_stopwords:
            old = self.stop_words.copy()
            self.stop_words = self.stop_words.difference(set(without_stopwords))
            if self.verbose: print(f"Removed {len(old)-len(self.stop_words)} stopword(s).")
            return self.stop_words

    def untokenize(self, text: List[str])-> str:
        """Revert tokenization: list of strings -> string"""
        return " ".join([w for w in text])

    def count_stopwords(self):
        print(f'{len(self.stop_words)} used.')
 
    def remove_whitespace(self, text : str)-> str:
        """Remove whitespaces"""
        return  " ".join(text.split())

    def remove_punctuation(self, text: str)-> str:    
       return [re.sub(f"[{re.escape(punctuation)}]", "", token) for token in text]

    def remove_numbers(self, text: str)-> str:    
       return [re.sub(r"\b[0-9]+\b\s*", "", token) for token in text]

    def remove_stopwords(self, text : str)-> str:
        return [token for token in text if token not in self.stop_words]

    def remove_digits(self, text: str)-> str: 
        """Remove digits instead of any number, e.g. keep dates"""
        return [token for token in text if not token.isdigit()]

    def remove_non_alphabetic(self, text: str)-> str: 
        """Remove non-alphabetic characters"""
        return [token for token in text if token.isalpha()]
    
    def remove_spec_char_punct(self, text: str)-> str: 
        """Remove all special characters and punctuation"""
        return [re.sub(r"[^A-Za-z0-9\s]+", "", token) for token in text]

    def remove_short_tokens(self, text: str, token_length : int = 2)-> str: 
        """Remove short tokens"""
        return [token for token in text if len(token) > token_length]

    def remove_punct(self, text: str)-> str:
        """Remove punctuations"""
        tokenizer = RegexpTokenizer(r"\w+")
        lst = tokenizer.tokenize(' '.join(text))
        # table = str.maketrans('', '', string.punctuation)          # punctuation
        # lst = [w.translate(table) for w in text]     # Umlaute
        return lst

    def replace_umlaut(self, text : str) -> str:
        """Replace special German umlauts (vowel mutations) from text"""
        vowel_char_map = {ord(k): v for k,v in self.umlaut['replace']['german']['umlaute'].items()}  # use unicode value of Umlaut
        return [token.translate(vowel_char_map) for token in text]

    def stem(self, text : str)-> str:
        """Apply nltk stemming"""
        return [self.stemmer.stem(w)  for w in text]
    
    def lemmatize(self, text : str)-> str:
        """Apply spaCy lemmatization"""
        text = self.untokenize(text)
        return [token.lemma_ for token in self.nlp(text)]

    def fit(self, X : pd.DataFrame, y : pd.Series = None):
        return self    
    
    def transform(self, X : pd.Series, **param)-> pd.Series:    
        corpus = deepcopy(X)
        if self.verbose: print("Setting to lower cases.")
        corpus = corpus.str.lower()
        if self.verbose: print("Removing whitespaces.")
        corpus = corpus.apply(self.remove_whitespace)
        if self.verbose: print("Applying word tokenizer.")
        corpus = corpus.apply(lambda x: word_tokenize(x))
        if self.verbose: print("Removing custom stopwords.") 
        corpus = corpus.apply(self.remove_stopwords)
        if self.verbose: print("Removing punctuations.")
        corpus = corpus.apply(self.remove_punct)
        if self.verbose: print("Removing numbers.")
        corpus = corpus.apply(self.remove_numbers)
        if self.verbose: print("Removing digits.")
        corpus = corpus.apply(self.remove_digits)
        if self.verbose: print("Removing non-alphabetic characters.")
        corpus = corpus.apply(self.remove_non_alphabetic)
        if self.verbose: print("Replacing German Umlaute.") 
        corpus = corpus.apply(self.replace_umlaut)  
        if self.verbose: print("Removing special character punctuations.")
        corpus = corpus.apply(self.remove_spec_char_punct)
        if self.verbose: print("Removing short tokens.")
        corpus = corpus.apply(self.remove_short_tokens, token_length=3)
        if self.stem: 
            if self.verbose: print("Applying stemming.") 
            corpus = corpus.apply(self.stem)          # German stemmer
        if self.lemma: 
            if self.verbose: print("Applying lemmatization.") 
            corpus = corpus.apply(self.lemmatize)  # makes preprocessing very slow though
        corpus = corpus.apply(self.untokenize)
        if self.verbose: print("Finished preprocessing.")
        return corpus #.to_frame(name="text") 


def get_document_embeddings(model : Callable, sentences : List[List[Text]], documents : List[Text])-> np.array:
    """
    Calculate Word2Vec document embeddings by simple arith. average of word embeddings.

    Args:
        model (Callable): _description_
        sentences (List[List[Text]]): _description_
        documents (List[Text]): _description_

    Returns:
        np.array: _description_
    """ 
    doc_vectors = np.empty((len(sentences),model.vector_size))
    index2doc, doc2index = {}, {}
    for i, doc in enumerate(tqdm(sentences, total=len(sentences))):
        index2doc[i], doc2index[str(documents[i])] = documents[i], i
        vec=np.empty((model.vector_size,0))    # empty column
        if len(doc)>0:
            for token in doc:
                vector = model.wv[token]
                vec = np.column_stack((vec, vector))
            doc_vectors[i,:] = np.nanmean(vec, axis=1)
    return doc_vectors, index2doc, doc2index


class word2vec_callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self, verbose : bool = True):
        self.epoch = 0
        self.verbose = verbose
        self.saved_loss_values = []
        print("Starting Word2Vec training loss monitoring...")

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()      # Note: this outputs the total, i.e. cumulative loss, until iteration t 
        if self.epoch == 0:
            if self.verbose: print('Loss - epoch {}: {}'.format(self.epoch, loss))
        else:
            if self.verbose and (self.epoch % 10 == 0): 
                print('Loss - epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss
        self.saved_loss_values.append(loss)