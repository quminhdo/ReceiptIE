from .estimatorA import EstimatorA
from .estimatorB import EstimatorB

def get_estimator(name, 
                net,
                criterion,
                optimizer,
                device,
                ckpt_dir):
    if name == "EstimatorA":
        return EstimatorA(net=net,
                        criterion=criterion,
                        optimizer=optimizer,
                        device=device,
                        ckpt_dir=ckpt_dir)
    if name == "EstimatorB":
        return EstimatorB(net=net,
                        criterion=criterion,
                        optimizer=optimizer,
                        device=device,
                        ckpt_dir=ckpt_dir)