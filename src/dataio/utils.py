import re
# import datefinder
from nltk.tokenize import word_tokenize

def is_date(text):
    pattern = r"\d{2,4}[.-/]\d{2}[.-/]\d{2,4}"
    m = re.search(pattern, text)
    return m is not None

def is_number(word):
    try:
        float(word)
    except:
        return False
    return True

def get_word_type(word):
    if is_date(word):
        return "DATE"
    if is_number(word):
        return "NUMBER"
    if word.isalpha():
        return "ALPHA"
    return "OTHER"

def find_nth(text, ss, n):
    inilist = [m.start() for m in re.finditer(ss, text)] 
    if len(inilist)>= n: 
        return inilist[n-1]
    else:
        raise "Exception: no nth occurence of %s in %s"%(ss, text)

def get_word_from_text(text):
    tokens = word_tokenize(text)
    valid_tokens = []
    for token in tokens:
        word_type = get_word_type(token)
        if word_type != "OTHER":
            valid_tokens.append(token)
    if len(valid_tokens) == 1:
        return valid_tokens[0]
    else:
        return text