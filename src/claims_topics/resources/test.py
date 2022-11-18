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

from claims_topics.utils import utils as util
from claims_topics.services import file
from claims_topics.config import global_config as glob
from pathlib import Path

pd.set_option('display.max_colwidth', 30)
pd.set_option('display.max_rows', 500)

reload(glob)
reload(util)
reload(file)

js = file.JSONservice(root_path=glob.UC_CODE_DIR + '/claims_topics/config', path='test.json')

german_stopwords = js.doRead()
german_stopwords

file_name = "Subsample.csv" #"Claim descr.csv"

df = file.CSVService(path=file_name, root_path=glob.UC_DATA_DIR, delimiter=",").doRead()

print(df.shape)


# Columns to use:
#-----------------
#col_sel = ['id_sch','invoice_item_id', 'dl_gewerk','firma', 'yylobbez', 'erartbez', 'hsp_eigen', 'hsp_prodbez', 'sartbez', 'sursbez', 'schilderung', 'de1_eks_postext']
col_sel = ['dl_gewerk','de1_eks_postext']
#col_sel = ['de1_eks_postext']

corpus = df[col_sel].drop_duplicates(subset=col_sel, keep=False)#.head(1*10**5)

print(corpus.shape)
corpus.head()
