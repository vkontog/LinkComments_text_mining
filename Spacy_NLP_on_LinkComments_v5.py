#!/usr/bin/env python
# coding: utf-8
# %%

# https://prrao87.github.io/blog/spacy/nlp/performance/2020/05/02/spacy-multiprocess.html
# 
# 

# Uncommend and run the following pip & python commands when running a new compute for the <b> first</b> time! 

# %%


# #!python -m spacy download el_core_news_sm
# #!pip install pyarrow --upgrade
# #!pip install openpyxl
# #!pip install xlrd


# %%


import spacy
#import el_core_news_sm
import string
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from azureml.core import Experiment
from azureml.core import Workspace, Dataset
from spacy.cli.download import download as spacy_download



# %%
from azureml.core import Run
run = Run.get_context(allow_offline=False)
ws = run.experiment.workspace
#ws = Workspace(subscription_id, resource_group, workspace_name)


# %%
spacy_download('el_core_news_sm')
nlp =spacy.load('el_core_news_sm', disable=['tagger', 'parser', 'ner'])


# %%


p1 = re.compile('δεν απαντ.{1,3}\s{0,1}',re.IGNORECASE)
p2 = re.compile('\sδα\s',re.IGNORECASE)
p3 = re.compile('δε.{0,1}\s.{0,3}\s{0,1}βρ.{1,2}κ.\s{0,1}',re.IGNORECASE)
p4 = re.compile('[^\d]?\d{10}')
p5 = re.compile('[^\d]?\d{18}|[^\d]\d{20}')
p6 = re.compile('δε[ ν]{0,1} (επιθυμ[α-ω]{2,4}?|[ήη]θ[εέ]λ[α-ω]{1,3}?|θελ[α-ω]{1,4}|.{0,20}ενδιαφ[εέ]ρ[α-ω]{2,4})',re.IGNORECASE)
p7 = re.compile('δε[ ν]{0,1} (μπορ[α-ω]{2,5}|.εχει)',re.IGNORECASE)
p8 = re.compile('(δεν|μη).*διαθεσιμ[οη]ς{0,1}?',re.IGNORECASE)
p9 = re.compile('(δεν|μη)+.*εφικτη?',re.IGNORECASE)
p10 = re.compile('δε[ ν]{0,1}.{0,20}θετικ[οόήη]ς{0,1}',re.IGNORECASE)


# %%


def loadStopWords():
    dataset = Dataset.get_by_name(ws, name='stopWords_gr')
    sw = set(dataset.to_pandas_dataframe())
    return sw


# %%


def replaceTerm(text):
    text = p5.sub(' λογαριασμός ',text)
    text = p4.sub(' τηλεφωνο ',text)
    text = p6.sub(' δενθελειδενενδιαφερεται ',text)
    text = p10.sub(' δενθελειδενενδιαφερεται ',text)
    text = p7.sub(' δενεχειδενμπορει ',text)
    text = p8.sub(' δενειναιδιαθεσιμος ',text)
    text = p9.sub(' ανεφικτη ',text)
    text = text.replace('-banking','banking')
    text = text.replace('v banking','vbanking')
    text = text.replace('e banking','ebanking')
    text = text.replace('follow up','followup')
    text = text.replace('fup','followup')
    text = text.replace('f/up','followup')
    text = text.replace('πυρ/ριο','πυρασφαλιστηριο')
    text = text.replace('safe drive','safedrive')
    text = text.replace('safe pocket','safepocket')
    text = text.replace('alphabank','alpha')
    text = text.replace('sweet home smart','sweethomesmart')
    text = text.replace('sweet home','sweethome')
    text = text.replace('eξασφαλιζω','εξασφαλιζω')
    text = text.replace('credit card','creditcard')
    text = text.replace('debit card','debitcard')
    text = text.replace('life cycle','lifecycle')
    text = text.replace('π/κ','πκ')
    text = text.replace('td','πκ')
    text = text.replace('α/κ','ακ')
    text = text.replace('δ/α','δεναπαντα ')
    text = text.replace('εκτος αττικης','εκτοςαττικης ')
    #τδ
    text = p1.sub(' δεναπαντα ',text)
    text = p2.sub(' δεναπαντα ',text)
    text = p3.sub(' δεντονβρηκα ',text)
    
    return text


# %%


sw = loadStopWords()
def remove_ton(text):
    diction = {'ά':'α','έ':'ε','ί':'ι','ό':'ο','ώ':'ω','ύ':'υ'}
    for key in diction.keys():
        text = text.replace(key, diction[key])
    return text   
def clean_text(text):
     #text to string
    text = str(text).lower()
    text = replaceTerm(text)
    
   # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # lower text
    text = [remove_ton(x) for x in text]
    # remove stop words
    text = [x for x in text if x not in sw]
 
    #remove quotes
    text = [x.replace('quot;','').replace('&quot','') for x in text if x not in {'quot','amp'}]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # remove amp & quot
    text = [x for x in text if x not in ['quot','amp']]
    # remove words with only one letter
    text = " ".join([t for t in text if len(t) > 1])
     # lemmatize text
    text = " ".join([t.lemma_ for t in nlp(text, disable=['tagger', 'parser', 'ner','tok2vec', 'morphologizer', 'parser', 'senter', 'attribute_ruler',  'ner'])])
   
    return(text)


# %%


def load_correctDict():
        
    dataset = Dataset.get_by_name(ws, name='correct_Tokens')    
    corDict = dict(dataset.to_pandas_dataframe().to_dict("split")['data'])
    return corDict


# %%


def correct(x,corDict):

    if x in corDict.keys():
        y = corDict[x]
    else:
        y = x
    return y    


# %%


def get_ngrams(idf,mindf,minngram,maxngram):
    tfidf = TfidfVectorizer(min_df = mindf,ngram_range = (minngram,maxngram))
    tfidf_result = tfidf.fit_transform(idf['tokenized']).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
    tfidf_df.columns = [str(x) for x in tfidf_df.columns]
    df_i = pd.concat([df[['CON_ROW_ID']],tfidf_df],axis=1).melt(id_vars=['CON_ROW_ID'],value_vars = tfidf_df.columns).dropna()
    df_i = df_i[df_i['value']>0]
    return df_i


# %%


def cleanComments(df):
    df = df[['CON_ROW_ID','CON_COMMENTS']]
    df['tokenized'] = df['CON_COMMENTS'].apply(clean_text)
    df = df.fillna('N/A')
    df['variable'] = df['tokenized'].str.split()
    return df


# %%


def getTokens(df):
    df = cleanComments(df)
    df_f = df.explode('variable')[['CON_ROW_ID','variable']]
    return df_f


# %%


def getTokencount(df_f,minCount):
    tokenCount = df_f['variable'].value_counts().to_dict()
    df_f['value'] = df_f['variable'].map(tokenCount)
    df_f=df_f[df_f['value']>=minCount] 
    return df_f


# %%


def loadComments(fileNum):
    # azureml-core of version 1.0.72 or higher is required
    # azureml-dataprep[pandas] of version 1.1.34 or higher is required
   


    dataset = Dataset.get_by_name(ws, name='LinkComments{0}'.format(fileNum))
    df = dataset.to_pandas_dataframe()
    return df


# %%


#experiment = Experiment(workspace = ws, name = "Link_Comments")


# %%

#run = Run.get_context()
#run = experiment.start_logging(snapshot_directory=None)


# %%


fileNum = '202105'


# %%


run.log('fileNum',fileNum)


# %%


df = loadComments(fileNum)


# %%


df = cleanComments(df)


# %%


df_f = getTokens(df)


# %%


minCount = 30


# %%


run.log('minCount',minCount)


# %%



df_f = getTokencount(df_f,minCount)


# %%


#ngrams parameters
mindf,minngram,maxngram = 1000,2,3


# %%


run.log_table('Parameters',{'Param':['mindf','minngram','maxngram'],'Values':[mindf,minngram,maxngram]})


# %%


df_f = df_f.append(get_ngrams(df,mindf,minngram,maxngram ))


# %%


#df_tokenCount = pd.read_excel('tokenlist.xlsx',engine='openpyxl')


# %%


#df_f['variable'].value_counts().to_excel('./xlsx/tokenlistTotal.xlsx')


# %%


corDict = load_correctDict()


# %%


df_f['token'] = df_f['variable'].apply(lambda x : correct(x,corDict))


# %%


df_f = df_f[df_f['token'] !='rmv']


# %%


df_f = df_f[df_f['token'].str.len() >1]


# %%


#df_f['token'].value_counts().to_excel('tokenlist.xlsx')


# %%


df_f = df_f.fillna('N/A')


# %%


df_f = df_f.sort_values(['CON_ROW_ID','token'])


# %%


df_f = df_f[['CON_ROW_ID','token']].drop_duplicates()


# %%


df_f.to_csv('comments_tokens_{0}.txt'.format(fileNum),sep ='\t',line_terminator='\r\n',index = False)


# %%
datastore = ws.get_default_datastore()

# %%
import os 
from os.path import join as osjoin

fil = [os.getcwd()+'/'+'comments_tokens_{0}.txt'.format(fileNum)]

# %%
datastore.upload_files(fil, target_path='UI/NLP', overwrite=True, show_progress=True)

# %%


#df_f['token'].value_counts().to_excel('./xlsx/tokenlist_new.xlsx')


# %%


print('comments_tokens_{0}.txt'.format(fileNum))


# %%


run.log('Output','comments_tokens_{0}.txt'.format(fileNum))


# %%


#run.complete()

