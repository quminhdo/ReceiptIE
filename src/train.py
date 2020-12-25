import os
import torch
from torch.nn import BCELoss
from network import get_network
from optimizer import get_optimizer
from dataio import get_dataset
from estimator import get_estimator
from utils import json_file_to_pyobj, get_embeddings
from argparse import ArgumentParser
def main(cf):

    net_opt = cf.network
    data_opt = cf.data
    train_opt = cf.train
    eval_opt = cf.eval
    cuda = cf.cuda
    ckpt_dir = cf.ckpt_dir

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    if cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    embeddings = get_embeddings(vocab_file=data_opt.vocab_file, word2vec_file=data_opt.word2vec_file)

    net = get_network(opt=net_opt, embeddings=embeddings)
    net = net.to(device)
    criterion = BCELoss()
    optimizer = get_optimizer(train_opt.optimizer, net.parameters())

    train_set = get_dataset(data_opt, "train")
    val_set = get_dataset(data_opt, "val")

    estimator = get_estimator(name=cf.estimator,
                            net=net,
                            criterion=criterion,
                            optimizer=optimizer,
                            device=device,
                            ckpt_dir=ckpt_dir)

    print("Training...")
    estimator.train(train_set=train_set, 
                    val_set=val_set, 
                    epoch_num=train_opt.epoch_num,
                    batch_size=train_opt.batch_size,
                    patience=train_opt.patience,
                    thresholds=eval_opt.thresholds)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", help="json config file")
    args = parser.parse_args()
    config = json_file_to_pyobj(args.config)
    main(config)