import spacy
import re

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

def tokenizer(s): 
    return [w.text for w in nlp(clean_data(s))]

def clean_data(text):
    text = re.sub(r'[^A-Za-z0-9@]+', ' ', text)   
    return text.strip()

reg_lookup = {
    'Phone_number': '((((|\\d{3})([\\s\\-./]?))|((|\\+\\d)([\\s\\-./]))))(((\\()(\\d{3})(\\))[\\s\\-./]?)|((\\d{3})[\\s\\-./]?)?)(?<![ ])(?<!^)(\\d{3})([\\s\\-./]?)(\\d{4})(x[\\d]{1,5})?\\b',
    'SSN': '(\\d{2,3})[- ](\\d{2})[- ](\\d{4})',
    'CreditCardNumber': '\b((\\d{4}))-?\\s?\\d{4}-?\\s?\\d{4}-?\\s?\\d{4}|\\d{19}|\\d{18}|\\d{17}|\\d{16}|\\d{15}|\\d{14}|\\d{13}|\\d{12}|\\d{11}\b',
    'Email': '[a-z0-9]+([-+._][a-z0-9]+){0,2}@.*?(\\.(a(?:[cdefgilmnoqrstuwxz]|ero|(?:rp|si)a)|b(?:[abdefghijmnorstvwyz]iz)|c(?:[acdfghiklmnoruvxyz]|at|o(?:m|op))|d[ejkmoz]|e(?:[ceghrstu]|du)|f[ijkmor]|g(?:[abdefghilmnpqrstuwy]|ov)|h[kmnrtu]|i(?:[delmnoqrst]|n(?:fo|t))|j(?:[emop]|obs)|k[eghimnprwyz]|l[abcikrstuvy]|m(?:[acdeghklmnopqrstuvwxyz]|il|obi|useum)|n(?:[acefgilopruz]|ame|et)|o(?:m|rg)|p(?:[aefghklmnrstwy]|ro)|qa|r[eosuw]|s[abcdeghijklmnortuvyz]|t(?:[cdfghjklmnoprtvwz]|(?:rav)?el)|u[agkmsyz]|v[aceginu]|w[fs]|y[etu]|z[amw]|biz)\\b){1,2}',
    'Plates': '\\b(([A-Z0-9]{1,4}[ -•]?[A-Z0-9]{2,6}))\\b'
}