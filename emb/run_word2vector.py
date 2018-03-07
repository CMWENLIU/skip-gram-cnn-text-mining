from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
import os

data_sets = ['/home/xxliu10/bigdata/bbcready/allbbc.news',
             #'/home/xxliu10/bigdata/collectready/allmk.news',
             #'/home/xxliu10/bigdata/bbc_mk.news',
             #'/home/xxliu10/bigdata/tweets/weets1.txt.new',
             #'/home/xxliu10/bigdata/bbc_tweets.news']

for data in data_sets:
	head, filename = os.path.split(data)
	filename += '.vec200.txt'
	filepath = '/home/xxliu10/bigdata/'	+ filename
	sentences = LineSentence(data)
	model = Word2Vec(sentences, size=200, window=5, min_count=5, workers=4)
	model.wv.save_word2vec_format(filepath, binary=False)
	print(filename + ' has been finished!')
#sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
#model_20 = Word2Vec(sentences, size=20, window=5, min_count=1, workers=4)
#model_20.save('w2v_20')
#model_80 = Word2Vec(sentences, size=80, window=5, min_count=1, workers=4)
#model_80.save('w2v_80')

#model_200 = Word2Vec(sentences, size=200, window=5, min_count=1, workers=4)
#model_200.save('/home/xxliu10/bigdata/w2v_200')

'''
model = Word2Vec.load('/home/xxliu10/bigdata/w2v_200')
vector = model['business']  # get vector for word
print(vector, len(vector))
#print('Vocabulary size of model is: ', len(eval('model').vocab))
model.wv.save_word2vec_format('/home/xxliu10/bigdata/w2v_200.model.txt', binary=False)
'''
