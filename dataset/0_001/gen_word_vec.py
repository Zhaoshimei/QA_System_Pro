from gensim.models import Word2Vec
import json
from nltk.tokenize import sent_tokenize, word_tokenize

modelname = r"C:\Users\NY\Desktop\QA_LSTM\mymodel.model"


#三个
corpus1 = []
docs_corpus = []
with open(r"C:\Users\NY\Desktop\QA_LSTM\documents.json",
          'r') as docs:
    docs_corpus = json.load(docs)
    for i in range(len(docs_corpus)):
        for para in docs_corpus[i]['text']:
            sents = sent_tokenize(para)
            # print("sents",sents)
            for sent in sents:
                tokens = word_tokenize(sent)
                # print("tokens",tokens)
                corpus1.append(tokens)
print("len(corpus1)", len(corpus1))
print("corpus1[0]", corpus1[0])

corpus2 = []
train_corpus = []
with open(r"C:\Users\NY\Desktop\QA_LSTM\training.json",
          'r') as training:
    train_corpus = json.load(training)
    # print("len(train_corpus)",len(train_corpus))
    for i in range(len(train_corpus)):
        question = train_corpus[i]['question']
        answer = train_corpus[i]['text']
        processed_question = word_tokenize(question)
        corpus2.append(processed_question)
        processed_answer = word_tokenize(answer)
        corpus2.append(processed_answer)
print("len(corpus2)", len(corpus2))
print("corpus2[0]", corpus2[0])

# test_for_train = train_corpus[:10000]

dev_corpus = []
with open(r"C:\Users\NY\Desktop\QA_LSTM\devel.json", 'r') as devel:
    dev_corpus = json.load(devel)

# test_for_del = dev_corpus[:1000]

corpus3 = []
test_corpus = []
with open(r"C:\Users\NY\Desktop\QA_LSTM\testing.json", 'r') as test:
    test_corpus = json.load(test)
    for i in range(len(test_corpus)):
        question = test_corpus[i]['question']
        processed_question = word_tokenize(question)
        corpus3.append(processed_question)
print("len(corpus3)", len(corpus3))
print("corpus3[0]", corpus3[0])

print(len(corpus1)+len(corpus2)+len(corpus3))
model = Word2Vec(corpus1+corpus2+corpus3,min_count=1)
model.save(modelname)