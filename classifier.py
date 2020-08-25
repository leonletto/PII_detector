import torch
import torch.nn as nn
import torch.optim as optim
from util import tokenizer

class classifier():
    def __init__(self, model, device):
        
        self.model = model
        
        self.device = device   
        
        self.optimizer = optim.Adam(model.parameters(), lr = 0.001)
        
        self.criterion = nn.NLLLoss().to(device)
        
        self.model.to(device)
               

    def test(self, test_iterator):
        epoch_loss = 0
        epoch_acc = 0
        
        self.model.eval()
        
        with torch.no_grad():
        
            for batch in test_iterator:
                data = batch.Text.to(self.device)       
                label =  batch.Labels.to(self.device)
                
                predictions = self.model(data)
                loss = self.criterion(predictions,label)       
                values, indices = predictions.max(1)               
                correct = (indices == label).float()  
                        
                acc = correct.sum() / len(correct)        
        
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        
        test_loss = epoch_loss / len(test_iterator)
        test_acc = epoch_acc / len(test_iterator)
        
        print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')
        
        return test_loss
        
    def train(self, train_iterator, path = None):
        
        epoch_loss = 0
        epoch_acc = 0
        
        self.model.train()
        
        for batch in train_iterator:      
            self.optimizer.zero_grad()     
            
            data = batch.Text.to(self.device)
            label =  batch.Labels.to(self.device)
            predictions = self.model(data)
            
            loss = self.criterion(predictions,label)
            
            values, indices = predictions.max(1)
                    
            correct = (indices == label).float() 
            
            acc = correct.sum() / len(correct)
            
            loss.backward()
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        if path != None:
            self.save_model(path)
            
        return epoch_loss / len(train_iterator), epoch_acc / len(train_iterator)
    
    def predict(self, sentence, TEXT):

        self.model.eval()

        tokenized = tokenizer(sentence)
        
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        
        tensor = torch.LongTensor(indexed)
        
        tensor = tensor.to(self.device)        
        tensor = tensor.unsqueeze(1)
        
        prediction = self.model(tensor).cpu()
        
        value, index = prediction.max(1)   
                
        return index  
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

       
    def restore_model(self, path):      
        self.model.load_state_dict(torch.load(path))
