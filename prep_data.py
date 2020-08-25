import pandas as pd


def process_classification_data(input_file, output_file):
    data = pd.read_csv(input_file, error_bad_lines = False)
    for i in range(len(data)):
        if data.loc[i].Labels != "Address" and data.loc[i].Labels != "Name" and data.loc[i].Labels != "Plates" and data.loc[i].Labels != "None":
            data.drop([i],inplace = True)       
        elif data.loc[i].Labels == "Plates":
            data.at[i,'Labels'] = "None"
            
    data.reset_index(inplace = True, drop = True)        
    data.to_csv(output_file, index = False)        

      

process_classification_data('data/PIITrainLargeData.csv', './data/test_classification.csv')
process_classification_data('data/PIITrain15K.csv', './data/train_classification.csv')

