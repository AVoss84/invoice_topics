import numpy as np
from typing import (Dict, List, Text, Optional, Any, Callable, Union)
from gensim.models.callbacks import CallbackAny2Vec
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy
from tqdm import tqdm
#from tqdm.auto import tqdm
import pandas as pd


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


def get_weighted_document_embeddings(model : Callable, tf_idf_vec : pd.DataFrame, sentences : List[List[Text]], documents : List[Text])-> np.array:
    """
    Calculate Word2Vec document embeddings by tf-idf weighted arith. average of word embeddings.

    Args:
        model (Callable): _description_
        sentences (List[List[Text]]): _description_

    Returns:
        np.array: _description_
    """ 
    n, vec_size = len(sentences), model.wv.vectors.shape[1]
    doc_vecs = np.zeros((n,vec_size))
    index2doc, doc2index = {}, {}
    for i, doc in enumerate(tqdm(sentences, total=len(sentences))):
      index2doc[i], doc2index[str(documents[i])] = documents[i], i
      doc_i = 0
      for token in doc:
          weight = tf_idf_vec.loc[i,token] if token in tf_idf_vec.columns else 0
          doc_i += weight * model.wv[token]
      doc_vecs[i,:] = doc_i  
    return doc_vecs, index2doc, doc2index


def get_fasttext_document_embeddings(model : Callable, documents : List[Text])-> np.array:
    """Get FastText document embeddings.
    Args:
        model (Callable): tarined fasttext model
        documents (List[Text]): preprocessed corpus

    Returns:
        np.array: Array of dimension #corpus_size x #embedding_dimension
    """
    assert isinstance(model, fasttext.FastText._FastText), 'model must be trained fasttext instance!'
    svec = [model.get_sentence_vector(d) for d in tqdm(documents, total=len(documents))]
    sent_embed = np.vstack(svec)
    return sent_embed

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