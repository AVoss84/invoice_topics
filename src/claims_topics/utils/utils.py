import numpy as np
from typing import (Dict, List, Text, Optional, Any, Callable, Union)
from gensim.models.callbacks import CallbackAny2Vec
import fasttext, os
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy
from tqdm import tqdm
#from tqdm.auto import tqdm
from claims_topics.config import global_config as glob
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
    index2doc, doc2index, svec = {}, {}, []
    #svec = [model.get_sentence_vector(d) for d in tqdm(documents, total=len(documents))]
    for i, d in enumerate(tqdm(documents, total=len(documents))):
        index2doc[i], doc2index[str(documents[i])] = documents[i], i
        svec.append(model.get_sentence_vector(d))
    sent_embed = np.vstack(svec)
    return sent_embed, index2doc, doc2index

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



class adjstruct_generator:
    """
    Utility function to generate weighted undirected graph from text embeddings in networkX
    """
    @staticmethod
    def create(embeddings : np.ndarray, corpus : pd.Series, nof_trees : int = 30, metric : str = "euclidean", 
                    filename_index : str = "knn_index.ann", seed : int = None, save_files_to_dir : str = glob.UC_DATA_PKG_DIR,
                    doc2index = dict, k : int = 5, filename_adjlist : str = 'graph_adjlist.txt', 
                    verbose : bool = True, dist_thresh : Optional[float] = None, **param)-> list:

        """Generate indexing forest via random projections for fast approx. kNN (LSH) using Spotify's Annoy. 
           Using the indexer object we create an adjacency list as input for networkX graph generation functions.
           Note: for large datasets you don't want to compute the (potentially sparse) NxN adjacency matrix directly, 
           rather than just save the k-nearest neighbors per node.  

        Args:
            embeddings (np.ndarray): N x dim_embeddings, with N the number of documents in the corpus 
            corpus (pd.Series): preprocessed corpus
            nof_trees (int, optional): _description_. Defaults to 20.
            metric (str, optional): _description_. Defaults to "euclidean".
            filename_index (str, optional): _description_. Defaults to "knn_index.ann".
            seed (int, optional): _description_. Defaults to None.
            save_files_to_dir (str, optional): _description_. Defaults to glob.UC_DATA_PKG_DIR.
            doc2index (_type_, optional): see own FastText/Word2Vec functions for creating document embeddings. Defaults to dict.
            k (int, optional): _description_. Defaults to 5.
            filename_adjlist (str, optional): _description_. Defaults to 'graph_adjlist.txt'.
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            dist: list of kNN distances
            metric_thresh: threshold of metric to construct sparse adjacency matrix  
        """
        assert isinstance(embeddings, np.ndarray), "embeddings must be np.array object"
        assert isinstance(corpus, pd.Series), "corpus must be Series object"
        X = deepcopy(embeddings)
        dim_embedding = X.shape[1]
        filename_index_path = os.path.join(save_files_to_dir, filename_index)
        filename_adjlist_path = os.path.join(save_files_to_dir, filename_adjlist)
        if verbose : print(f"** Using {metric} metric for approx. {k}-NN **")
        #------------------------------------------
        # Create index forest for approx. kNN:
        #-------------------------------------------------------------------------
        knn_indexer = AnnoyIndex(dim_embedding, metric = metric, **param)
        if seed: knn_indexer.set_seed(seed)
        for i in tqdm(range(X.shape[0]), total=X.shape[0]):
            v = X[i,:].tolist()
            knn_indexer.add_item(i, v)
        #----------------------------------------------------------------------
        knn_indexer.build(nof_trees)                    
        try:
            os.stat(filename_index_path)  # file exists?
        except:
            file = open(filename_index_path, "w")   # else create empty file
            file.close()

        knn_indexer.save(filename_index_path)   # save index forest
        if verbose: print(f'- Indexing finished and saved as: {filename_index_path}')
        #------------------------
        # Create adjacancy list
        #--------------------------------------------------------------------------
        dist, i = [], 1; metric_thresh = dist_thresh if dist_thresh else 0 
        with open(filename_adjlist_path, 'w') as fp:
            for doc in tqdm(corpus.tolist(), total=len(corpus.tolist())):
                neigh = knn_indexer.get_nns_by_item(doc2index[doc], k) 
                for item in neigh[1:]:   
                    d = knn_indexer.get_distance(doc2index[doc], item)
                    if dist_thresh is None: metric_thresh = (metric_thresh*(i - 1) + d)/i                  
                    if np.round(d,5) > metric_thresh:               # only set edge between node (document) pair if threshold is exceeded!
                        fp.write(str(neigh[0]) +" "+ str(item)+" "+str(np.round(d,3)))   # with weights
                        fp.write("\n")
                        dist.append(d)
        if verbose: print(f'- Adjacency list created and saved as: {filename_adjlist_path}'); print('Done.')
        return dist, metric_thresh
        

