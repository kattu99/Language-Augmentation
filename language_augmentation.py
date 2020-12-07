from nltk.corpus import wordnet 
from nltk.corpus import stopwords  
import gensim.downloader as api
import time
import os

os.environ["GENSIM_DATA_DIR"] = "/Users/rahulkataria/Documents/RepresentationLearning/"

class LanguageAugmentation: 

    def __init__(self, vocab): 
        self.word_set = set()
        self.stop_words = set(stopwords.words('english'))  
        for tup in vocab:
            self.word_set.add(tup[0])
        self.model = api.load("glove-twitter-25")
        self.vocab = self.model.vocab

    def augment_sentence(self, tokens):
        for i in range(len(tokens)): 
            print(tokens[i])
            if tokens[i] not in self.stop_words:
                synonyms = self.get_synonyms(tokens[i])
                synonyms.sort(key=lambda x: (x[1], -x[2]))
                print(synonyms)
                if len(synonyms) > 0: 
                    for synonym in synonyms:
                        if synonym[0] != tokens[i]:
                            tokens[i] = synonym[0]
                            break
        return tokens 

    def get_synonyms(self,word):
        synonyms = []
        syns = wordnet.synsets(word)
        second_syns = self.model.most_similar(word)
        start = time.time()
        for syn in syns: 
            for l in syn.lemmas():
                # check the model entry 
                if l.name() in self.vocab and word in self.vocab:
                    # check the VL-BERT word set
                    if l.name() in self.word_set: 
                        synonyms.append((l.name(), 1, self.model.similarity(l.name(), word)))
                    else: 
                        synonyms.append((l.name(), 0, self.model.similarity(l.name(), word)))
                else: 
                    if l.name() in self.word_set: 
                        synonyms.append((l.name(), 1, 0))
                    else: 
                        synonyms.append((l.name(), 0, 0))
        
        for second in second_syns:
            if second in self.vocab and word in self.vocab:
                if second in self.word_set: 
                    synonyms.append((second, 0, self.model.similarity(second, word)))
                else: 
                    synonyms.append((second, 0, self.model.similarity(second, word)))
            else: 
                if second in self.word_set: 
                    synonyms.append((second, 0, 0))
                else: 
                    synonyms.append((second, 0, 0))

        end = time.time()
        print(end - start)
        return synonyms

