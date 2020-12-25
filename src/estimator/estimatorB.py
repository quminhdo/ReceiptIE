import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score, f1_score

class EstimatorB:

    def __init__(self,
                net,
                criterion,
                optimizer,
                device,
                ckpt_dir):
        # self.train_set = train_set
        # self.val_set = val_set
        # self.test_set = test_set
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.ckpt_dir = ckpt_dir

    def save(self, path):
        torch.save(self.net.state_dict(), path)
        print("Save model to %s"%path)

    def train(self, train_set, val_set, epoch_num, batch_size, patience, thresholds):
        best_loss = 9999999
        best_epoch = 0
        no_improvement_count = 0

        for epoch in range(1, epoch_num + 1):
            print("\nEpoch %d:"%(epoch))
            self.train_one_epoch(train_set=train_set, batch_size=batch_size)
            self.save(os.path.join(self.ckpt_dir, "ckpt_%02d.pt"%epoch))

            print("Validation results")
            with torch.no_grad():
                val_loss = self.evaluate(test_set=val_set, batch_size=batch_size, thresholds=thresholds)
            # for param_group in self.optimizer.param_groups:
            #     print("Learning rate:", param_group['lr'], "\n")
            # self.scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if no_improvement_count == patience:
                break

    def train_one_epoch(self, train_set, batch_size):
        total_loss = 0.
        batch_num = 0
        self.net.train()
        for i in range(len(train_set)):
            page_data = train_set.get_single_page_data(i)
            for field_type, cand_data in page_data.items():
                inputs = {}
                for k in ["field_embed", "cand_pos", "neighbor_embed", "neighbor_pos"]:
                    inputs[k] = cand_data[k].to(self.device)
                labels = cand_data["label"].to(self.device)
                j = 0
                while j < len(labels):
                    batch_num += 1
                    self.optimizer.zero_grad()
                    batch_inputs = {k: v[j:j+batch_size] for k, v in inputs.items()}
                    batch_labels = labels[j:j+batch_size]
                    batch_scores = self.net(batch_inputs)
                    loss = self.criterion(batch_scores, batch_labels)
                    loss.backward()
                    self.optimizer.step()
                    # print(loss.item())
                    total_loss += loss.item()
                    j += batch_size
        mean_loss = total_loss/batch_num
        print("Training loss: %.6f"%mean_loss)

    def evaluate(self, test_set, batch_size, thresholds):
        total_loss = 0.
        batch_num = 0
        y_true = {"DATE": [], "TOTAL": []}
        y_score = {"DATE": [], "TOTAL": []}
        y_pred = {"DATE": {str(k): [] for k in thresholds}, "TOTAL": {str(k): [] for k in thresholds}}
        roc_auc = {}
        f1 = {"DATE": {}, "TOTAL": {}}

        self.net.eval()
        for i in range(len(test_set)):
            page_data = test_set.get_single_page_data(i)
            for field_type, cand_data in page_data.items():
                inputs = {}
                # for k in ["field_id", "cand_pos", "neighbor_id", "neighbor_pos"]:
                #     inputs[k] = torch.from_numpy(cand_data[k]).to(self.device)
                # labels = torch.from_numpy(cand_data["label"]).to(self.device)
                for k in ["field_embed", "cand_pos", "neighbor_embed", "neighbor_pos"]:
                    inputs[k] = cand_data[k].to(self.device)
                labels = cand_data["label"].to(self.device)
                j = 0
                while j < len(labels):
                    batch_num += 1
                    batch_inputs = {k: v[j:j+batch_size] for k, v in inputs.items()}
                    batch_labels = labels[j:j+batch_size]
                    batch_scores = self.net(batch_inputs)
                    loss = self.criterion(batch_scores, batch_labels)
                    # print(loss.item())
                    total_loss += loss.item()
                    j += batch_size

                    # accumulate results
                    assert batch_labels.ndim == 1 and batch_scores.ndim == 1
                    y_true[field_type].append(batch_labels.int().detach().cpu().numpy())
                    y_score[field_type].append(batch_scores.detach().cpu().numpy())
                    for thresh in thresholds:
                        batch_preds = (batch_scores >= thresh).int()
                        y_pred[field_type][str(thresh)].append(batch_preds.detach().cpu().numpy())

        mean_loss = total_loss/batch_num
        for field_type in ["DATE", "TOTAL"]:
            # concat results
            y_true[field_type] = np.concatenate(y_true[field_type], axis=0)
            y_score[field_type] = np.concatenate(y_score[field_type], axis=0)

            max_f1 = 0
            best_thresh = 0
            for thresh in thresholds:
                y_pred[field_type][str(thresh)] = np.concatenate(y_pred[field_type][str(thresh)], axis=0)
                f1[field_type][str(thresh)] = f1_score(y_true[field_type], y_pred[field_type][str(thresh)])
                if f1[field_type][str(thresh)] >= max_f1:
                    max_f1 = f1[field_type][str(thresh)]
                    best_thresh = thresh
            f1[field_type]["max"] = max_f1
            f1[field_type]["best_thresh"] = best_thresh
            roc_auc[field_type] = roc_auc_score(y_true[field_type], y_score[field_type])
        self.print_report(mean_loss, f1, roc_auc)
        return mean_loss
    
    def print_report(self, loss, f1, roc_auc):
        print("Loss: %.6f"%loss)
        print("ROC AUC:")
        print("\tdate: %.6f"%roc_auc["DATE"])
        print("\ttotal: %.6f"%roc_auc["TOTAL"])
        print("max F1:")
        print("\tdate: %.6f at threshold %.2f"%(f1["DATE"]["max"], f1["DATE"]["best_thresh"]))
        print("\ttotal: %.6f at threshold %.2f"%(f1["TOTAL"]["max"], f1["TOTAL"]["best_thresh"]))