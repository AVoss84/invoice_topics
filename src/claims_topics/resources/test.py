import numpy as np
import pandas as pd
#import networkx as nx
import os, gensim
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder 
from umap import UMAP
from copy import deepcopy
from importlib import reload

from claims_topics.utils import utils
from claims_topics.resources import preprocessor as preproc
from claims_topics.services import file
from claims_topics.config import global_config as glob
from claims_topics.config import config
from pathlib import Path

pd.set_option('display.max_colwidth', 30)
pd.set_option('display.max_rows', 500)

reload(glob)
reload(utils)
reload(file)
reload(config)
reload(preproc)
#-------------------------------------------------------------------------------------
config_input = config.io['input']

df = file.CSVService(root_path=glob.UC_DATA_DIR, **config_input['service']['CSVService']).doRead()

print(df.shape)

#-------------------------------------------------------------------------------------
# Columns to use:
#-----------------
#col_sel = ['id_sch','invoice_item_id', 'dl_gewerk','firma', 'yylobbez', 'erartbez', 'hsp_eigen', 'hsp_prodbez', 'sartbez', 'sursbez', 'schilderung', 'de1_eks_postext']
col_sel = ['dl_gewerk','de1_eks_postext']
#col_sel = ['de1_eks_postext']

corpus = df[col_sel].drop_duplicates(subset=col_sel, keep=False)#.head(1*10**5)

print(corpus.shape)
corpus.head()

#-------------------------------------------------------------------------------------
# Create labels for supervised topic modeling:
target = LabelEncoder().fit_transform(corpus['dl_gewerk'].tolist())   # labels

# Build corpus
X = corpus['de1_eks_postext']
#corpus['target'] = target

# Preprocess corpus:
cleaner = preproc.clean_text(language='german', without_stopwords=['nicht', 'keine'], lemma = False, stem = True)

X_cl = cleaner.fit_transform(X)

docs = X_cl.tolist()                            # format for BertTopic
target_names = corpus['dl_gewerk'].tolist()       # class labels

corpus_cl = X_cl.apply(lambda x: word_tokenize(x))       # this format needed for word2vec training only

sentences = corpus_cl.tolist() 

#-------------------------------------------------------------------------------------
model = Word2Vec.load(glob.UC_DATA_DIR + "/Word2Vec_embeddings.model")

#-------------------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

#vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words=cleaner.stop_words)
X = vectorizer.fit_transform(docs)
tf_idf_vec = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out().tolist())
tf_idf_vec.head()


#-------------------------------------------------------------------------------------
reload(utils)

# Call:
#embeddings, index2doc, doc2index = utils.get_document_embeddings(model, sentences, docs)
embeddings, index2doc, doc2index = utils.get_weighted_document_embeddings(model, tf_idf_vec, sentences, docs)

print(embeddings.shape)