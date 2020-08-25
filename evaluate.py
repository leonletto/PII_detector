import pandas as pd
import re
from util import reg_lookup 
from address_name_classifier import address_name_classifier
from address_name_chunker import address_name_chunker

LOAD_MODEL = False

if __name__ == '__main__':
    input_file = 'data/PIITrain15K.csv' 
    output_file = 'data/PIITrain15K_Evaluate.csv'
    
    data = pd.read_csv(input_file, error_bad_lines = False)
    data['Labels_found'] = data.Labels.astype(str)
    data['PII_found'] = data.Labels.astype(str)   

    add_n_classifier = address_name_classifier(LOAD_MODEL)
    add_n_chunker = address_name_chunker(LOAD_MODEL)

              
    for i in range(len(data)):
        if i % 1000 == 0:
            print(f'{i}/{len(data)} completed.')
            
        phone = re.search(reg_lookup['Phone_number'], data.loc[i].Text)
        ssn = re.search(reg_lookup['SSN'], data.loc[i].Text)
        creditcardnumber = re.search(reg_lookup['CreditCardNumber'], data.loc[i].Text)
        email = re.search(reg_lookup['Email'], data.loc[i].Text)
        plate = re.search(reg_lookup['Plates'], data.loc[i].Text)
        
        if email:
            data.at[i,'Labels_found'] = 'Email'
            data.at[i,'PII_found'] = email.group(0).strip()
            
        elif ssn:
            data.at[i,'Labels_found'] = 'SSN'
            data.at[i,'PII_found'] = ssn.group(0).strip()
            
        elif creditcardnumber:
            data.at[i,'Labels_found'] = 'CreditCardNumber'
            data.at[i,'PII_found'] = creditcardnumber.group(0).strip()
    
        elif phone:
            data.at[i,'Labels_found'] = 'Phone_number'
            data.at[i,'PII_found'] = phone.group(0).strip()
               
        else:
            prediction = add_n_classifier.predict(data.loc[i].Text)  
            
            if add_n_classifier.is_address(prediction):
                data.at[i,'Labels_found'] = 'Address'
                data.at[i,'PII_found'] = add_n_chunker.get_address(data.loc[i].Text)
           
            elif plate:
                data.at[i,'Labels_found'] = 'Plates'
                data.at[i,'PII_found']= plate.group(0).strip()
                    
            elif add_n_classifier.is_name(prediction):       
                data.at[i,'Labels_found'] = 'Name'
                data.at[i,'PII_found'] = add_n_chunker.get_name(data.loc[i].Text)
            
            else:
                data.at[i,'Labels_found'] = 'None'
                data.at[i,'PII_found'] = 'None'
                       
        
    data.to_csv(output_file, index = False)  
    
    # =MID(A2,IF(FIND(C2,A2,1)-10<1,1,FIND(C2,A2,1)-10),IF(LEN(C2)+20>LEN(A2)-FIND(C2,A2,1)+11,LEN(A2)-FIND(C2,A2,1)+11,LEN(C2)+20))