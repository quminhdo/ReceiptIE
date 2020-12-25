from .moduleA import NeuralScoringModelA
from .moduleB import NeuralScoringModelB

def get_network(opt, embeddings):

    if opt.name == "NeuralScoringModelA":
        return NeuralScoringModelA(d=opt.hidden_size,
                                    field_embeddings=embeddings,
                                    text_embeddings=embeddings,
                                    attention_num=opt.attention_num,
                                    dropout_rate=opt.dropout)
    if opt.name == "NeuralScoringModelB":
        return NeuralScoringModelB(d=opt.hidden_size,
                                    embedding_size=opt.embedding_size,
                                    attention_num=opt.attention_num,
                                    dropout_rate=opt.dropout)