import pandas as pd
from nltk import word_tokenize, pos_tag
import pickle

O ='O'
I ='I-GPE'
B ='B-GPE'

def pii_locator(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll] == sl:
            return ind , ind+sll
        
        elif sl[-1] == '.' and l[ind+sll-1] != '.':
             sl_t = sl[:-1]
             sl_t[-1] = sl_t[-1] + '.'
             if l[ind:ind+sll-1] == sl_t:
                 return ind , ind+sll-1
        

def iob_tag_data(input_file, output_file, label):
    data = pd.read_csv(input_file, error_bad_lines = False)  
    iob_tagged_data = []
    
    for i in range(len(data)):
        if data.loc[i].Labels == label:
            text = word_tokenize(data.loc[i].Text)
            pii = word_tokenize(data.loc[i].PII)
            
            start, end = pii_locator(pii, text)
            
            pos_text = pos_tag(text)
            
            iob_tagged_text = []
            
            for j in range(len(pos_text)):
                if j == start:
                   iob_tagged_text.append((pos_text[j],B)) 
                elif j in range(start + 1, end):                
                   iob_tagged_text.append((pos_text[j],I)) 
                else:
                   iob_tagged_text.append((pos_text[j],O)) 
    
            iob_tagged_data.append(iob_tagged_text)
     
    pickle_out = open(output_file,"wb")
    pickle.dump(iob_tagged_data, pickle_out)
    pickle_out.close()
           

iob_tag_data('./data/PIITrainLargeData.csv', './data/iob_tagged_test_address.pkl', 'Address')
iob_tag_data('./data/PIITrain15K.csv', './data/iob_tagged_train_address.pkl', "Address")

iob_tag_data('./data/PIITrainLargeData.csv', './data/iob_tagged_test_name.pkl', "Name")
iob_tag_data('./data/PIITrain15K.csv', './data/iob_tagged_train_name.pkl', "Name")
