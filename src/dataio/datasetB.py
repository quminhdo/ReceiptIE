from collections import Counter, defaultdict
from glob import glob
import json
import random
import re
import os
import cv2
import torch
import numpy as np
from flair.embeddings import WordEmbeddings, BytePairEmbeddings, StackedEmbeddings
from flair.data import Sentence
from .utils import find_nth, get_word_type, get_word_from_text, is_number, stack
from .box import Box, Component, get_intersection_box

class OCR_Word(Box):

    def __init__(self, info):
        sep = find_nth(info, ",", 8)
        self.text = get_word_from_text(info[sep+1:]).lower()
        self.x1, self.y1, _, _, self.x2, self.y2, _, _ = [int(x) for x in info[:sep].split(",")]
        self.id = None

    def resize_randomly(self, page_width, page_height, ratio):
        new_x1 = int(self.x1 + (2*random.random()-1)*self.w*ratio)
        new_x2 = int(self.x2 + (2*random.random()-1)*self.w*ratio)
        new_y1 = int(self.y1 + (2*random.random()-1)*self.h*ratio)
        new_y2 = int(self.y2 + (2*random.random()-1)*self.h*ratio)
        self.x1 = max(0, new_x1)
        self.x2 = min(new_x2, page_width)
        self.y1 = max(0, new_y1)
        self.y2 = min(new_y2, page_height)


class Neighbor(Component):

    def __init__(self,
                text,
                x1,
                y1,
                x2,
                y2,
                page_width,
                page_height,
                origin_coords):

        super(Neighbor, self).__init__(text=text, x1=x1, y1=y1, x2=x2, y2=y2,
                                        page_width=page_width, page_height=page_height)
        # if is_number(self.text):
        #     self.text = "[NUMBER]"
        self.xc0, self.yc0 = origin_coords

    @property
    def position(self):
        return self.norm_xc - self.xc0, self.norm_yc - self.yc0 
    
class Candidate(Component):

    def __init__(self,
                text,
                wtype,
                word_ids,
                words,
                page_width,
                page_height,
                max_neighbors):
        self.text = text
        self.wtype = wtype
        self.pw = page_width
        self.ph = page_height
        self.word_ids = word_ids
        self.max_neighbors = max_neighbors
        self.x1, self.y1, self.x2, self.y2 = self.generate_box(words, word_ids)
        self.neighborhood = self.get_neighborhood()
        self.neighbors = self.get_neighbors(words)

    def generate_box(self, words, word_ids):
        for i in word_ids:
            for j in word_ids:
                if words[i].y1 > words[j].y2:
                    raise "words (%d, %d, %d, %d) and (%d, %d, %d, %d) are not in the same line"%(words[i].x1, words[i].y1, words[i].x2, words[i].y2,
                                                                                                words[j].x1, words[j].y1, words[j].x2, words[j].y2)
        x1 = min([words[i].x1 for i in word_ids])
        y1 = min([words[i].y1 for i in word_ids])
        x2 = max([words[i].x2 for i in word_ids])
        y2 = max([words[i].y2 for i in word_ids])
        return x1, y1, x2, y2

    def get_neighborhood(self, h_ratio=0.1):
        x1 = 0
        y1 = max(0, int(self.y1 - h_ratio*self.ph))
        x2 = self.x2
        y2 = self.y2
        return Box(x1, y1, x2, y2)

    def get_neighbors(self, words):
        neighbors = []
        for word in words:
            if word.id in self.word_ids:
                continue
            inter = get_intersection_box(word, self.neighborhood)
            if inter.area >= word.area / 2:
                neighbors.append(Neighbor(text=word.text, x1=word.x1, y1=word.y1, x2=word.x2, y2=word.y2, 
                                        page_width=self.pw, page_height=self.ph, origin_coords=self.position))
        random.shuffle(neighbors)
        if len(neighbors) > self.max_neighbors:
            neighbors = neighbors[:self.max_neighbors]
        else:
            for _ in range(self.max_neighbors - len(neighbors)):
                x1 = random.choice(range(self.neighborhood.x1, self.neighborhood.x2))
                y1 = random.choice(range(self.neighborhood.y1, self.neighborhood.y2))
                x2 = random.choice(range(x1, self.neighborhood.x2))
                y2 = random.choice(range(y1, self.neighborhood.y2))
                neighbors.append(Neighbor(text="[PAD]", x1=x1, y1=y1, x2=x2, y2=y2, page_width=self.pw, page_height=self.ph, origin_coords=self.position))
        return neighbors

    @property
    def position(self):
        return self.norm_xc, self.norm_yc

class ReceiptDatasetB:

    def __init__(self, opt, split):
        file_names = self.get_file_names(os.path.join(opt.split_dir, "%s_list.txt"%split))
        self.ocr_files = [os.path.join(opt.ocr_data_dir, fn) for fn in file_names]
        self.ie_files = [os.path.join(opt.ie_data_dir, fn) for fn in file_names]
        self.max_neighbors = opt.max_neighbors
        self.word2id = self.get_word2id(opt.vocab_file)
        self.split = split

        self.embeddings = StackedEmbeddings(
            [
                # standard FastText word embeddings for English
                WordEmbeddings('en'),
                # Byte pair embeddings for English
                BytePairEmbeddings('en'),
            ]
        )

    def __len__(self):
        return len(self.ie_files)

    def generate_candidates(self, idx):
        words = self.get_ocr_words(idx)
        width, height = self.get_page_width_height(idx)
        for i, word in enumerate(words):
            word.id = i
            if self.split == "train":
                word.resize_randomly(width, height, 0.1)

        candidate_dict = {"DATE": [], "TOTAL": []}
        for word in words:
            wtype = get_word_type(word.text)
            if wtype in ["DATE", "NUMBER"]:
                if wtype == "DATE":
                    field_type = "DATE"
                if wtype == "NUMBER":
                    field_type = "TOTAL"
                candidate_dict[field_type].append(Candidate(text=word.text, 
                                                            wtype=wtype,
                                                            word_ids=[word.id],
                                                            words=words,
                                                            page_width=width,
                                                            page_height=height,
                                                            max_neighbors=self.max_neighbors))
        return candidate_dict

    def get_single_page_data(self, idx):
        with open("%s.txt"%self.ie_files[idx], "r", encoding="utf-8") as f:
            d = json.load(f)
        gt = {"DATE": d["date"], "TOTAL": d["total"]}

        candidate_dict = self.generate_candidates(idx)
        data = {}
        for field_type, cands in candidate_dict.items():
            field_embed_list = []
            cand_pos_list = []
            neighbor_embed_list = []
            neighbor_pos_list = []
            label_list = []
            random.shuffle(cands)
            for cand in cands:
                field_embed_list.append(self.word2vec(field_type.lower()))
                cand_pos_list.append(list(cand.position))
                neighbor_embed_list.append([self.word2vec(neighbor.text) for neighbor in cand.neighbors])
                neighbor_pos_list.append([list(neighbor.position) for neighbor in cand.neighbors])
                label_list.append(int(cand.text == gt[field_type]))
            data[field_type] = {"field_embed": stack(field_embed_list),
                                "cand_pos": torch.from_numpy(np.array(cand_pos_list, dtype=np.float32)),
                                "neighbor_embed": stack(neighbor_embed_list),
                                "neighbor_pos": torch.from_numpy(np.array(neighbor_pos_list, dtype=np.float32)),
                                "label": torch.from_numpy(np.array(label_list, dtype=np.float32))}
        return data

    def word2vec(self, word):
        sentence = Sentence(word, use_tokenizer=False)
        self.embeddings.embed(sentence)
        tok = sentence.tokens[0]
        return tok.embedding

    def get_page_width_height(self, idx):
        img = cv2.imread("%s.jpg"%self.ie_files[idx])
        height, width, _ = img.shape
        return width, height

    def get_ocr_words(self, idx):
        with open("%s.txt"%self.ocr_files[idx], "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        return [OCR_Word(l) for l in lines]

    def get_file_names(self, f):
        with open(f, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        return lines

    def get_word2id(self, vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            words = f.read().splitlines()
        words = {word : i for i, word in enumerate(words)}
        word2id = defaultdict(lambda: words["[UNK]"])
        word2id.update(words)
        return word2id