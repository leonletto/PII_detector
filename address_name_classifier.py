import torch
import torchtext
import os
from model import rnn_class
from classifier import classifier
from util import tokenizer

class address_name_classifier():
    
       def __init__(self, LOAD_MODEL = True):
           
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
            self.TEXT = torchtext.data.Field(tokenize = tokenizer)
            self.LABEL = torchtext.data.LabelField(dtype = torch.long)
            
            datafields = [('Text', self.TEXT),('Labels', self.LABEL)]
            
            trn, tst = torchtext.data.TabularDataset.splits(path = './data/', train = 'train_classification.csv', test = 'test_classification.csv',    
                                                            format = 'csv', skip_header = True, fields = datafields)
            
            self.TEXT.build_vocab(trn, max_size = 25000, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
            self.LABEL.build_vocab(trn)
                        
            # Model Hyperparameter
            input_dim = len(self.TEXT.vocab)
            embedding_dim = 100
            hidden_dim = 20
            output_dim = len(self.LABEL.vocab)
            n_layers = 2
            max_num_epochs = 20
            dropout = 0.5
            patience = 3
            model_name = "classifier.pt"
            path = os.path.join('saved_models', model_name)
                        
            model = rnn_class(input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
                        
            self.add_n_classifier = classifier(model, device)           

            if not LOAD_MODEL:
                epochs_no_improve = 0 
                pretrained_embeddings = self.TEXT.vocab.vectors
                model.embedding.weight.data.copy_(pretrained_embeddings)
                
                unk_idx = self.TEXT.vocab.stoi[self.TEXT.unk_token] #0
                pad_idx = self.TEXT.vocab.stoi[self.TEXT.pad_token] #1
                model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
                model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
                
                train_iterator, test_iterator = torchtext.data.BucketIterator.splits((trn, tst), batch_size = 16,
                                shuffle=True, sort_key=lambda x: len(x.Text), sort_within_batch = False)
                
                prev_test_error = None            
                epoch = 0
                while epoch < max_num_epochs:  
                    if epochs_no_improve == patience:
                        print('Early stopping!')
                        break
                    else:
                        train_loss, train_acc = self.add_n_classifier.train(train_iterator, path)   
                        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% |')       
                        epoch = epoch + 1
                    
                        test_error = self.add_n_classifier.test(test_iterator)  
                        if not prev_test_error:
                            prev_test_error  = test_error
                        elif test_error > prev_test_error:
                            epochs_no_improve = epochs_no_improve + 1                        
                        
                        prev_test_error  = test_error                        
            else:
                self.add_n_classifier.restore_model(path)    
                
            
       def predict(self, sentence):
           return self.add_n_classifier.predict(sentence, self.TEXT)
           
       def is_address(self, prediction):
            return (prediction == self.LABEL.vocab.stoi["Address"]).item() 
        
       def is_name(self, prediction):
            return (prediction== self.LABEL.vocab.stoi["Name"]).item() 
        
        