from .module import NeuralScoringModel

def get_network(opt, embeddings):
    net = NeuralScoringModel(d=opt.hidden_size,
                            field_embeddings=embeddings,
                            text_embeddings=embeddings,
                            attention_num=opt.attention_num,
                            dropout_rate=opt.dropout)

    return net