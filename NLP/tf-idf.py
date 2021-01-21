from collections import Counter
from tqdm import tqdm
from scipy.sparse import csr_matrix
import math
import operator
from sklearn.preprocessing import normalize
import numpy as np

class customized_tfidf_2():
    def __init__(self,corpus):
        self.corpus=corpus
        self.unique_words=[]
        self.lengthofcorpus=0
        self.idf_={}
    
    #Function to calculate Inverse-Document-Freq Dictionary
    def idf(self):
        assert len(self.unique_words)!=0,'Must run fit function prior to this'
        for val in list(self.unique_words.keys()):
            g=0
            for i in corpus:
                g+=int(val in i.split())
            self.idf_[val]=1+(math.log((1+len(self.corpus))/(1+g)))
        return dict(sorted(self.idf_.items(),key=lambda item:item[1],reverse=True)[:50])
    
    #Function to calculate tf-idf
    def fit(self):
        # temporary list
        temp1,size=[],len(self.unique_words)
        for i in self.corpus:
            self.lengthofcorpus+=len(i.split())
            temp1+=[i for i in i.split() if i not in temp1]
        self.unique_words={val:idx for idx,val in enumerate(temp1)}
        self.idf_of_unique_words=self.idf()
        for i in list(self.unique_words.keys()):
            if i not in list(self.idf_of_unique_words.keys()):
                self.unique_words.pop(i)
        return self.unique_words,self.idf_of_unique_words
        
        
    def transform(self,vocab,idf):
        sparse_matrix=csr_matrix((len(self.corpus), max(vocab.values())))
        for i in range(len(self.corpus)):
            number_of_words_in_sentence=Counter(self.corpus[i].split())
            for word in self.corpus[i].split():
                if word in list(vocab.keys()):
                    yf_idf_value=(number_of_words_in_sentence[word]/len(self.corpus[i].split()))*(idf[word])
                    sparse_matrix[i,vocab[word]-1]=yf_idf_value
        print('NORM FORM',normalize(sparse_matrix,norm='l2',axis=1))
        output=normalize(sparse_matrix,norm='l2',axis=1)
        return output
        
#Checking the result
vec=customized_tfidf_2(corpus)
vocab,idf=vec.fit()
