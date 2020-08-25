import torch
import torch.nn as nn

class rnn_class(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 label_dim, n_layers, dropout):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)        
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, 
                           bidirectional = True, dropout = dropout)    
        
        self.fc = nn.Linear(hidden_dim*2, label_dim)     
        
        self.softmax = nn.LogSoftmax(dim=1)               
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, text):
        
        embedded = self.dropout(self.embedding(text))
        
        output, hidden = self.rnn(embedded)
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
        hidden = self.fc(hidden)
        
        hidden = self.softmax(hidden)
        
        return hidden
