# coding: utf-8

# In[2]:


from __future__ import print_function
from functools import reduce
import re
import json

from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent, LSTM, Dropout, Merge, Dense
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences

# In[9]:
import os
import nltk
import string
from math import log
from nltk.corpus import wordnet as wn
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, RepeatVector
import gc


lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


docs_corpus = []
with open(r"C:\Users\NY\Desktop\QA_LSTM\documents.json", 'r') as docs:
    docs_corpus = json.load(docs)

train_corpus = []
with open(r"C:\Users\NY\Desktop\QA_LSTM\training.json", 'r') as training:
    train_corpus = json.load(training)
print("len(train_corpus)", len(train_corpus))

test_for_train = train_corpus[:2000]

dev_corpus = []
with open(r"C:\Users\NY\Desktop\QA_LSTM\devel.json", 'r') as devel:
    dev_corpus = json.load(devel)

test_for_del = dev_corpus[:200]

punc = string.punctuation
stopwordsAll = set(stopwords.words('english'))

def qestion_and_answer(question_id):
    question = train_corpus[question_id]['question']
    answer = train_corpus[question_id]['text']
    processed_question = word_tokenize(question)
    processed_answer = word_tokenize(answer)
    para_id = train_corpus[question_id]['answer_paragraph']
    docid = train_corpus[question_id]['docid']
    # print("processed_question",processed_question)
    # print("answer",answer)
    # print("docid",docid)
    return processed_question, processed_answer, para_id,docid


# qestion_and_answer(0)

def doc_to_story(para_id,docid):
    story = []

    doc = docs_corpus[docid]
    para = ""
    for index, para_data in enumerate(doc['text']):
        if index == para_id:
            para = para_data
        sents = sent_tokenize(para)
        # print("sents",sents)
        for sent in sents:
            tokens = word_tokenize(sent)
            # print("tokens",tokens)
            story.append(tokens)
    # print("story",story)
    return story


# doc_to_story(0)

# save [(story, question, answer)]
def prepare_data(train_corpus):
    final_data = []
    print(len(train_corpus))
    for i in range(len(train_corpus)):
        processed_question, answer, para_id,docid = qestion_and_answer(i)
        story = doc_to_story(para_id,docid)
        final_data.append((story, processed_question, answer))
        # print("final_data",final_data)
    return final_data




'''
def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma
'''


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        story_list = []
        for sent in story:
            for w in sent:
                story_list.append(w)
        x = [word_idx[w] for w in story_list]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        for token in answer:
            y[word_idx[token]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen), np.array(ys)


def get_vocab(data,del_data):
    vocab = set()
    for story, q, answer in data:
        list_words = []
        for sent in story:
            list_words = list_words + sent
        vocab_list = list_words + q + answer
        # print("vocab_list",vocab_list)
        vocab |= set(vocab_list)
    for story, q, answer in del_data:
        list_words = []
        for sent in story:
            list_words = list_words + sent
        vocab_list = list_words + q + answer
        # print("vocab_list",vocab_list)
        vocab |= set(vocab_list)
    vocab = sorted(vocab)
    return vocab

data = prepare_data(test_for_train)
print("len(data)", len(data))


def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
    return lemma


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        story_list = []
        for sent in story:
            for w in sent:
                story_list.append(w)
        x = [word_idx[w] for w in story_list]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        for w in answer:
            y[word_idx[w]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen), np.array(ys)


def get_paragraph(docid, document_data):
    # get the paragraph that contains the answer
    for item in document_data:
        if item['docid'] == docid:
            document = item['text']
            break
    return document


def term_freqs(document):
    tfs = defaultdict(dict)
    tfs_forward = defaultdict(dict)
    doc_id = 0
    for sentence in document:
        for token in word_tokenize(sentence):
            if token not in stopwordsAll and token not in punc:
                term = lemmatize(token.lower())
                tfs[term][doc_id] = tfs[term].get(doc_id, 0) + 1
                tfs_forward[doc_id][term] = tfs[doc_id].get(term, 0) + 1
        doc_id += 1
    return tfs, doc_id + 1, tfs_forward


def get_okapibm25(tf, total_docment, documents):
    '''Calculate and return term weights based on okapibm25'''
    k1, b, k3 = 1.5, 0.5, 0
    okapibm25 = defaultdict(dict)

    # calculate average doc length
    total = 0
    for d in documents:
        total += len(d)
    avg_doc_length = total / len(documents) * 1.0

    for term, doc_list in tf.items():
        df = len(doc_list)
        for doc_id, freq in doc_list.items():
            # term occurences in query
            # qtf = question.count(term) # SEPCIAL
            qtf = 1.2
            idf = log((total_docment - df + 0.5) / df + 0.5)
            tf_Dt = ((k1 + 1) * tf[term][doc_id]) / (
            k1 * ((1 - b) + b * (len(documents[doc_id]) / avg_doc_length) + tf[term][doc_id]))
            if qtf == 0:
                third = 0
            else:
                third = ((k3 + 1) * qtf) / (k3 + qtf)
                okapibm25[term][doc_id] = idf * tf_Dt * third

    return okapibm25


# find top_k paragraph that may contain the answer
def get_top_k_document(tfidf, query, k, document):
    top_document_id = Counter()
    for token in word_tokenize(query):
        if token not in stopwordsAll:
            term = lemmatizer.lemmatize(token.lower())
            term_tfidf = tfidf[term]
            for docid, weight in term_tfidf.items():
                top_document_id[docid] += weight
    top_document_id = top_document_id.most_common(k)
    top_document = []
    for document_id, weight in top_document_id:
        top_document.append(document_id)
    return top_document

def prepare_test_del(train_corpus):
    final_data = []
    print(len(train_corpus))
    for i in range(len(train_corpus)):
        story = []
        question = train_corpus[i]['question']
        processed_question, answer, para_id, docid = qestion_and_answer(i)
        document = get_paragraph(docid, docs_corpus)
        tfs, total_docment, tfs_forward = term_freqs(document)
        tfidf = get_okapibm25(tfs, total_docment, document)
        top_1 = get_top_k_document(tfidf, question, 1, document)
        for item in top_1:
            # print("item", item)
            story = doc_to_story(item, docid)
            # print("story",story)
        final_data.append((story, processed_question, answer))
        # print("final_data",final_data)
    return final_data

del_data = prepare_test_del(test_for_del)
print("len(del_data)", len(del_data))

vocab = get_vocab(data,del_data)
# print("vocab",vocab)
print("len(vocab)", len(vocab))

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
WORD2VEC_EMBED_SIZE = 100
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40

WORD2VEC_EMBED_SIZE = 100
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE,
                                                           SENT_HIDDEN_SIZE,
                                                           QUERY_HIDDEN_SIZE))

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word2idx = dict((c, i + 1) for i, c in enumerate(vocab))
print("len(word2idx)",len(word2idx))
story_maxlen = max(map(len, (x for x, _, _ in data)))
query_maxlen = max(map(len, (x for _, x, _ in data)))

x, xq, y = vectorize_stories(data, word2idx, story_maxlen, query_maxlen)
tx, txq, ty = vectorize_stories(del_data, word2idx, story_maxlen, query_maxlen)

del data
del del_data
gc.collect()

#word2vec
modelname = r"C:\Users\NY\Desktop\QA_LSTM\mymodel.model"
model = Word2Vec.load(modelname)  # 3个文件放在一起：Word60.model   Word60.model.syn0.npy   Word60.model.syn1neg.npy
print("read model successful")
embedding_weights = np.zeros((vocab_size, WORD2VEC_EMBED_SIZE))
for word, index in word2idx.items():
    try:
        embedding_weights[index, :] = model[word.lower()]
    except KeyError:
        pass  # keep as zero (not ideal, but what else can we do?)

del model
del word2idx
gc.collect()







#former



# print('vocab = {}'.format(vocab))
print('x.shape = {}'.format(x.shape))
print('xq.shape = {}'.format(xq.shape))
print('y.shape = {}'.format(y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print('Build model...')
'''model 
sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
encoded_sentence = layers.Embedding(vocab_size, WORD2VEC_EMBED_SIZE,
                                    weights=[embedding_weights],mask_zero=True)(sentence)
#encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_sentence)
encoded_sentence = layers.Dropout(0.3)(encoded_sentence)


question = layers.Input(shape=(query_maxlen,), dtype='int32')
encoded_question = layers.Embedding(vocab_size, WORD2VEC_EMBED_SIZE,
                                    weights=[embedding_weights],mask_zero=True)(question)
encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
encoded_question = layers.Dropout(0.3)(encoded_question)
encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)

merged = layers.add([encoded_sentence, encoded_question])
merged = RNN(EMBED_HIDDEN_SIZE)(merged)
merged = layers.Dropout(0.3)(merged)
preds = layers.Dense(vocab_size, activation='softmax')(merged)

model = Model([sentence, question], preds)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''
'''
sequential
'''
qenc = Sequential()
qenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   weights=[embedding_weights], mask_zero=True))
qenc.add(LSTM(EMBED_HIDDEN_SIZE, input_shape=(None,story_maxlen), return_sequences=False))
qenc.add(Dropout(0.3))

aenc = Sequential()
aenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   weights=[embedding_weights], mask_zero=True))
aenc.add(LSTM(EMBED_HIDDEN_SIZE, input_shape=(None,query_maxlen), return_sequences=False))
#aenc.add(RepeatVector(story_maxlen))
aenc.add(Dropout(0.3))

model = Sequential()
model.add(Merge([qenc, aenc], mode="sum"))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])



print('Training')
MODEL_DIR = r"C:\Users\NY\Desktop\QA_LSTM"
checkpoint = ModelCheckpoint(
    filepath = os.path.join(MODEL_DIR, "qa-lstm-best.hdf5"),
    verbose=1, save_best_only=True)
model.fit([x, xq], y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05)
loss, acc = model.evaluate([tx, txq], ty,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

