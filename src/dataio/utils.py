import re
# import datefinder
from nltk.tokenize import word_tokenize
import torch
import numpy as np

def is_date(text):
    pattern = r"\d{1,4}[.\-/](\d{1,2}|[a-zA-Z]{3})[.\-/]\d{2,4}"
    m = re.search(pattern, text)
    return m is not None

def is_currency(word):
    matches = re.findall(r"-{0,1}\d+\.\d+", word)
    if len(matches) != 1:
        return False
    if float(matches[0]) <= 0:
        return False
    return True

def is_number(word):
    try:
        float(word)
    except:
        return False
    return True

def get_word_type(word):
    if is_date(word):
        return "DATE"
    if is_currency(word):
        return "CURRENCY"
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

def stack(T):
    if isinstance(T, torch.Tensor):
        return T
    elif isinstance(T, np.ndarray):
        return T
    elif isinstance(T, list):
        if len(T) == 0:
            return torch.from_numpy(np.array([], dtype=np.float32))
        t_list = []
        for t in T:
            t_list.append(stack(t))
        if isinstance(t_list[0], torch.Tensor):
            return torch.stack(t_list)
        if isinstance(t_list[0], np.ndarray):
            return np.stack(t_list)
    else:
        raise Exception("%s data type is not supported"%type(T))