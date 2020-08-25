from chunker import chunker_factory

address_model = "./saved_models/address_chunker.pickle"
name_model = "./saved_models/name_chunker.pickle"

class address_name_chunker():
    
       def __init__(self, LOAD_MODEL = True):
           
          if not LOAD_MODEL:
              
              self.address_chunker = chunker_factory.create_address_chunker('data/iob_tagged_train_address.pkl')
              self.name_chunker = chunker_factory.create_name_chunker('data/iob_tagged_train_name.pkl')
              
              test_acc = chunker_factory.evaluate_chunker(self.address_chunker,'data/iob_tagged_test_address.pkl')
              train_acc = chunker_factory.evaluate_chunker(self.address_chunker,'data/iob_tagged_train_address.pkl')
              print(f'| Address Chunker | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% |')
              
              test_acc = chunker_factory.evaluate_chunker(self.name_chunker,'data/iob_tagged_test_name.pkl')
              train_acc = chunker_factory.evaluate_chunker(self.name_chunker,'data/iob_tagged_train_name.pkl')
              print(f'| Name Chunker | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% |')
              
              self.address_chunker.save_to_file(address_model)
              self.name_chunker.save_to_file(name_model)
        
          else:            
              self.address_chunker = chunker_factory.load_chunker(address_model)
              self.name_chunker = chunker_factory.load_chunker(name_model)

       def get_address(self, sentence):
            return self.address_chunker.chunk(sentence)
            
       def get_name(self, sentence):
            return self.name_chunker.chunk(sentence)
