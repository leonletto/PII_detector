import pickle
import string
from nltk import pos_tag, word_tokenize
from nltk.chunk import ChunkParserI, conlltags2tree
from nltk.tag import ClassifierBasedTagger 
from nltk.tag.util import untag
from nltk.stem.snowball import SnowballStemmer

class chunker_factory():
    
    @staticmethod
    def create_address_chunker(train_samples):
        with open(train_samples, 'rb') as fp:
            dataset = pickle.load(fp)
            chunker = address_chunker(dataset)    
        return chunker
    
    @staticmethod
    def create_name_chunker(train_samples):
        with open(train_samples, 'rb') as fp:
            dataset = pickle.load(fp)
            chunker = name_chunker(dataset)    
        return chunker
    
    def load_chunker(file_name):
        chunker_f = open(file_name, "rb")
        chunker = pickle.load(chunker_f)
        chunker_f.close()        
        return chunker

    @staticmethod    
    def evaluate_chunker(chunker, test_samples):  
        accuracy = 0         
        with open(test_samples, 'rb') as fp: 
            
            dataset = pickle.load(fp)
            for i in range(len(dataset)):
                 
                score = chunker.evaluate( [ conlltags2tree( [(w, t, iob) for ((w, t), iob) in dataset[i]]) ] )                
                accuracy = accuracy + score.accuracy()
            
        return accuracy / len(dataset) 
    

class address_chunker(ChunkParserI):
    
    def __init__(self, train_sents, **kwargs):
        self.tagger = ClassifierBasedTagger(
            train = train_sents,
            feature_detector = self.features,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
        return conlltags2tree(iob_triplets)
    
    def features(self, tokens, index, history):
        # for more details see: http://nlpforhackers.io/named-entity-extraction/ 
        
        """
        `tokens`  = a POS-tagged sentence [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        `history` = the previous predicted IOB tags
        """

        # init the stemmer
        stemmer = SnowballStemmer('english')

        # Pad the sequence with placeholders
        tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
        history = ['[START2]', '[START1]'] + list(history)

        # shift the index with 2, to accommodate the padding
        index += 2

        word, pos = tokens[index]
        prevword, prevpos = tokens[index - 1]
        prevprevword, prevprevpos = tokens[index - 2]
        nextword, nextpos = tokens[index + 1]
        nextnextword, nextnextpos = tokens[index + 2]
        previob = history[index - 1]
        contains_dash = '-' in word
        contains_dot = '.' in word
        allascii = all([True for c in word if c in string.ascii_lowercase])

        allcaps = word == word.capitalize()
        capitalized = word[0] in string.ascii_uppercase

        prevallcaps = prevword == prevword.capitalize()
        prevcapitalized = prevword[0] in string.ascii_uppercase

        nextallcaps = nextword == nextword.capitalize()
        nextcapitalized = nextword[0] in string.ascii_uppercase

        return {
            'word': word,
            'lemma': stemmer.stem(word),
            'pos': pos,
            'all-ascii': allascii,

            'next-word': nextword,
            'next-lemma': stemmer.stem(nextword),
            'next-pos': nextpos,

            'next-next-word': nextnextword,
            'next-next-pos': nextnextpos,

            'prev-word': prevword,
            'prev-lemma': stemmer.stem(prevword),
            'prev-pos': prevpos,

            'prev-prev-word': prevprevword,
            'prev-prev-pos': prevprevpos,

            'prev-iob': previob,

            'contains-dash': contains_dash,
            'contains-dot': contains_dot,

            'all-caps': allcaps,
            'capitalized': capitalized,

            'prev-all-caps': prevallcaps,
            'prev-capitalized': prevcapitalized,

            'next-all-caps': nextallcaps,
            'next-capitalized': nextcapitalized,
        }

    
    def save_to_file(self, file_name):
        save_classifier = open(file_name,"wb")
        pickle.dump(self, save_classifier)
        save_classifier.close()

    def chunk(self, sentence):
    
        tagged_tree = self.parse(pos_tag(word_tokenize(sentence)))
        
        chunks = []      
        for subtree in tagged_tree.subtrees(filter = tree_filter):
            chunks.append(untag(subtree.leaves()))          
        
        max_length = 0   
        for i in range(len(chunks)):
            if len(chunks[i]) > max_length:
                chunk = chunks[i]
                max_length = len(chunks[i])
                               
        output =''
        if len(chunks) > 0:
            for i in range(len(chunk)):
                if not chunk[i] == '.' and not chunk[i] == ',' and not i == 0:
                    output = output + ' ' + chunk[i]
                else:
                   output = output + chunk[i] 
                    
        return output
            

class name_chunker(ChunkParserI):
    
    def __init__(self, train_sents, **kwargs):
        self.tagger = ClassifierBasedTagger(
            train = train_sents,
            feature_detector = self.features,
            **kwargs)
        
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
        return conlltags2tree(iob_triplets)
        
    def save_to_file(self, file_name):
        save_classifier = open(file_name,"wb")
        pickle.dump(self, save_classifier)
        save_classifier.close()
        
    
    def features(self, tokens, index, history):
        # for more details see: http://nlpforhackers.io/named-entity-extraction/ 
        
        """
        `tokens`  = a POS-tagged sentence [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        `history` = the previous predicted IOB tags
        """
        
        # init the stemmer
        stemmer = SnowballStemmer('english')

        # Pad the sequence with placeholders
        tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
        history = ['[START2]', '[START1]'] + list(history)

        # shift the index with 2, to accommodate the padding
        index += 2

        word, pos = tokens[index]
        prevword, prevpos = tokens[index - 1]
        prevprevword, prevprevpos = tokens[index - 2]
        nextword, nextpos = tokens[index + 1]
        nextnextword, nextnextpos = tokens[index + 2]

        previob = history[index - 1]
 

        return {
            'word': word,
            'pos': pos,
            'lemma': stemmer.stem(word),

            'next-word': nextword,
            'next-pos': nextpos,
            'next-lemma': stemmer.stem(nextword),

            'next-next-lemma': stemmer.stem(nextnextword),

            'prev-word': prevword,
            'prev-pos': prevpos,
            'prev-lemma': stemmer.stem(prevword),

            'prev-iob': previob,

        }

    
    def chunk(self, sentence):
    
        tagged_tree = self.parse(pos_tag(word_tokenize(sentence.lower())))
        
        chunks = []      
        for subtree in tagged_tree.subtrees(filter = tree_filter):
            chunks.append(untag(subtree.leaves()))          
        
        max_length = 0   
        for i in range(len(chunks)):
            if len(chunks[i]) > max_length:
                chunk = chunks[i]
                max_length = len(chunks[i])
         
        output =''
        if len(chunks) > 0:
                        
            for i in range(len(chunk)):
                if not chunk[i] == '.' and not chunk[i] == ',' and not i == 0:
                    output = output + ' ' + chunk[i]
                else:
                   output = output + chunk[i] 
                   
            index = sentence.lower().find(output)  
            output = sentence[index:len(output)+index]
        
        return output
    
    
def tree_filter(tree):
    return tree.label() == "GPE"


