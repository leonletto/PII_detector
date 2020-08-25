import pandas as pd
import re
from util import reg_lookup 
from address_name_classifier import address_name_classifier
from address_name_chunker import address_name_chunker

LOAD_MODEL = True

if __name__ == '__main__':
    
    input_file = 'data/PIITestData.csv' 
    output_file = 'data/PIITestData_Output.csv'
    
    data = pd.read_csv(input_file, error_bad_lines = False)
    data['Label'] = data.Label.astype(str)
    data['PII'] = data.Label.astype(str)   

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
            data.at[i,'Label'] = 'Email'
            data.at[i,'PII'] = email.group(0).strip()
            
        elif ssn:
            data.at[i,'Label'] = 'SSN'
            data.at[i,'PII'] = ssn.group(0).strip()
            
        elif creditcardnumber:
            data.at[i,'Label'] = 'CreditCardNumber'
            data.at[i,'PII'] = creditcardnumber.group(0).strip()
    
        elif phone:
            data.at[i,'Label'] = 'Phone_number'
            data.at[i,'PII'] = phone.group(0).strip()
               
        else:
            prediction = add_n_classifier.predict(data.loc[i].Text)  
            
            if add_n_classifier.is_address(prediction):
                data.at[i,'Label'] = 'Address'
                data.at[i,'PII'] = add_n_chunker.get_address(data.loc[i].Text)
           
            elif plate:
                data.at[i,'Label'] = 'Plates'
                data.at[i,'PII']= plate.group(0).strip()
                    
            elif add_n_classifier.is_name(prediction):       
                data.at[i,'Label'] = 'Name'
                data.at[i,'PII'] = add_n_chunker.get_name(data.loc[i].Text)
            
            else:
                data.at[i,'Label'] = 'None'
                data.at[i,'PII'] = 'None'

        
    data.to_csv(output_file, index = False)  
