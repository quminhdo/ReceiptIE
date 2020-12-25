import torch
from torch.nn import Module, Sequential, Linear, Embedding, Dropout, ReLU, CosineSimilarity
from .layers import NeighborEncoder, Similarity

class NeuralScoringModelA(Module):

    def __init__(self, 
                d,
                field_embeddings,
                text_embeddings,
                attention_num,
                dropout_rate):
        super(NeuralScoringModelA, self).__init__()
        self.field_embedding = Sequential(Embedding.from_pretrained(field_embeddings), Linear(text_embeddings.size()[1], d))
        self.cand_position_embedding = Linear(2, d)
        self.text_embedding = Sequential(Embedding.from_pretrained(text_embeddings), Linear(text_embeddings.size()[1], d))
        self.neighbor_position_embedding = Sequential(Linear(2, d),
                                                    ReLU(),
                                                    Dropout(p=dropout_rate),
                                                    Linear(d, d),
                                                    ReLU(),
                                                    Dropout(p=dropout_rate))
        self.neighbor_encoder = NeighborEncoder(d, attention_num)
        self.candidate_encoder = Sequential(Linear(3*d, d),
                                            ReLU())
        self.scorer = Similarity()

    def forward(self, inputs):
        field_id = inputs["field_id"]
        cand_pos = inputs["cand_pos"]
        neighbor_id = inputs["neighbor_id"]
        neighbor_pos = inputs["neighbor_pos"]

        field_embed = self.field_embedding(field_id)
        cand_pos_embed = self.cand_position_embedding(cand_pos)
        text_embed = self.text_embedding(neighbor_id)
        pos_embed = self.neighbor_position_embedding(neighbor_pos)

        H = torch.cat((text_embed, pos_embed), dim=-1)
        neighbor_encoding = self.neighbor_encoder(H)
        neighborhood_encoding = torch.max(neighbor_encoding, dim=1).values
        # print(cand_pos_embed.size(), neighborhood_encoding.size())
        x = torch.cat((cand_pos_embed, neighborhood_encoding), dim=-1)
        candidate_encoding = self.candidate_encoder(x)
        score = self.scorer(field_embed, candidate_encoding)
        return score