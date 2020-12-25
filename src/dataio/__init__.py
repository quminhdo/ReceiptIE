from collections import Counter, defaultdict
from .datasetA import ReceiptDatasetA
from .datasetB import ReceiptDatasetB

def get_dataset(opt, split):
    
    if opt.dataset == "ReceiptDatasetA":
        return ReceiptDatasetA(opt, split)
    if opt.dataset == "ReceiptDatasetB":
        return ReceiptDatasetB(opt, split)