import os
from collections import OrderedDict
from operator import __add__
from pathlib import Path
from typing import Callable, List, Union
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.utils.data import Dataset
import math
from sklearn import metrics
from tqdm import tqdm
import random


def sigmoid(x):
    y = 1/(1+math.exp(-1*x))
    return(y)

def bootstrap(x,y):
    youden = 0.07800596181964313
    opt_threshold = youden
    y_total, yhat_total = x,y # y_total is ground truth, while yhat_total is prediction
    fpr_boot = []
    tpr_boot = []
    aucs = []
    sensitivity = []
    specificity = []
    ppv = []
    npv = []
    num_pos = []
    # bootstrap for confidence interval
    for i in tqdm(range(0,10000)):
        
        choices = list(yhat_total.sample(frac = 0.5).index)
                
        ground_sample = y_total[choices]
        preds_sample = yhat_total[choices]
        
        fpr,tpr, _ = metrics.roc_curve(ground_sample, preds_sample)
        fpr_boot.append(fpr)
        tpr_boot.append(tpr)
        aucs.append(metrics.auc(fpr,tpr))
        tp = len(ground_sample[(ground_sample == 1) & (preds_sample > opt_threshold)])
        fp = len(ground_sample[(ground_sample == 0) & (preds_sample > opt_threshold)])
        tn = len(ground_sample[(ground_sample == 0) & (preds_sample < opt_threshold)])
        fn = len(ground_sample[(ground_sample == 1) & (preds_sample < opt_threshold)])
        num_positives = len(ground_sample[(preds_sample > opt_threshold)])
        sensitivity.append(tp/(tp+fn))
        specificity.append(tn/(tn+fp))
        ppv.append(tp/(tp+fp))
        npv.append(tn/(tn+fn))
        num_pos.append(num_positives)

    np.array(aucs)
    np.array(ppv)
    np.array(npv)
    np.array(sensitivity)
    np.array(specificity)
    np.array(num_pos)
    
    fpr,tpr, _ = metrics.roc_curve(y_total, yhat_total)
    auc_full = metrics.auc(fpr,tpr)
    tp = len(y_total[(y_total == 1) & (yhat_total > opt_threshold)])
    fp = len(y_total[(y_total == 0) & (yhat_total > opt_threshold)])
    tn = len(y_total[(y_total == 0) & (yhat_total < opt_threshold)])
    fn = len(y_total[(y_total == 1) & (yhat_total < opt_threshold)])
    sensitivity_full = (tp/(tp+fn))
    specificity_full = (tn/(tn+fp))
    ppv_full = (tp/(tp+fp))
    npv_full = (tn/(tn+fn))

    auc_boot = [round(np.percentile(aucs,2.5),3), round(np.percentile(aucs,97.5),3)]
    sensitivity_boot = [round(np.percentile(sensitivity,2.5),3), round(np.percentile(sensitivity,97.5),3)]
    specificity_boot = [round(np.percentile(specificity,2.5),3), round(np.percentile(specificity,97.5),3)]
    ppv_boot = [round(np.percentile(ppv,2.5),3), round(np.percentile(ppv,97.5),3)]
    npv_boot = [round(np.percentile(npv,2.5),3), round(np.percentile(npv,97.5),3)]
    num_positives = [round(np.percentile(num_pos,2.5),3), round(np.percentile(num_pos,97.5),3)]
    
    
    print("auc:",round(auc_full,3),str(auc_boot))
    print("sensitivity:",round(sensitivity_full,3),str(sensitivity_boot))
    print("specificity:", round(specificity_full,3), str(specificity_boot))
    print("ppv:",round(ppv_full,3),str(ppv_boot))
    print("npv:",round(npv_full,3),str(npv_boot))
    # print("Number of Positives" + str(num_positives))

def bootstrap_reg_new(manifest):
    threshold = 35
    tp = len(manifest[(manifest.EF_2D < threshold) & (manifest.EF_preds < threshold)])
    fp = len(manifest[(manifest.EF_2D >= threshold) & (manifest.EF_preds < threshold)])
    fn = len(manifest[(manifest.EF_2D < threshold) & (manifest.EF_preds >= threshold)])
    tn = len(manifest[(manifest.EF_2D >= threshold) & (manifest.EF_preds >= threshold)])
    ppv_full = tp/(tp+fp)
    npv_full = tn/(fn+tn)
    sensitivity_full = tp/(fn+tp)
    specificity_full = tn/(tn+fp)
    fpr,tpr,_ = metrics.roc_curve((manifest.EF_2D >= threshold)*1,manifest.EF_preds)
    auc_full = metrics.auc(fpr,tpr)

    auc_list = []
    sens_list = []
    spec_list = []
    ppv_list = []
    npv_list = []

    for i in tqdm(range(0,10000)):
        predictions = manifest.sample(frac = 0.5)
        tp = len(predictions[(predictions.EF_2D < threshold) & (predictions.EF_preds < threshold)])
        fp = len(predictions[(predictions.EF_2D >= threshold) & (predictions.EF_preds < threshold)])
        fn = len(predictions[(predictions.EF_2D < threshold) & (predictions.EF_preds >= threshold)])
        tn = len(predictions[(predictions.EF_2D >= threshold) & (predictions.EF_preds >= threshold)])
        fpr,tpr,_ = metrics.roc_curve((predictions.EF_2D >= threshold)*1,predictions.EF_preds)
        auc = metrics.auc(fpr,tpr)
        auc_list.append(auc)
        sens_list.append(tp/(tp+fn))
        spec_list.append(tn/(tn+fp))
        ppv_list.append(tp/(tp+fp))
        npv_list.append(tn/(tn+fn))

    auc_boot = [round(np.percentile(auc_list,2.5),3), round(np.percentile(auc_list,97.5),3)]
    sensitivity_boot = [round(np.percentile(sens_list,2.5),3), round(np.percentile(sens_list,97.5),3)]
    specificity_boot = [round(np.percentile(spec_list,2.5),3), round(np.percentile(spec_list,97.5),3)]
    ppv_boot = [round(np.percentile(ppv_list,2.5),3), round(np.percentile(ppv_list,97.5),3)]
    npv_boot = [round(np.percentile(npv_list,2.5),3), round(np.percentile(npv_list,97.5),3)]

    print('AUC is ' + str(round(auc_full,3)) + str() + ' ' + str(auc_boot))
    print('Sensitivity is ' + str(round(sensitivity_full,3)) + ' ' + str(sensitivity_boot))
    print('Specificity is ' + str(round(specificity_full,3)) + ' ' + str(specificity_boot))
    print('PPV is ' + str(round(ppv_full,3)) + str() + ' ' + str(ppv_boot))
    print('NPV is ' + str(round(npv_full,3)) + str() + ' ' + str(npv_boot))

class EffNet(nn.Module):

    # lightly retouched version of John's EffNet to add clean support for multiple output
    # layer designs as well as single-lead inputs
    def __init__(
        self,
        num_extra_inputs: int = 0,
        output_neurons: int = 1,
        channels: List[int] = (32, 16, 24, 40, 80, 112, 192, 320, 1280),
        depth: List[int] = (1, 2, 2, 3, 3, 3, 3),
        dilation: int = 2,
        stride: int = 8,
        expansion: int = 6,
        embedding_hook: bool = False,
        input_channels: int = 12,
        verbose: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.channels = channels
        self.output_nerons = output_neurons

        # backwards compatibility change to prevent the addition of the output_neurons param
        # from breaking people's existing EffNet initializations
        if len(self.channels) == 10:
            self.output_nerons = self.channels[9]
            print(
                "DEPRECATION WARNING: instead of controlling the number of output neurons by changing the 10th item in the channels parameter, use the new output_neurons parameter instead."
            )

        self.depth = depth
        self.expansion = expansion
        self.stride = stride
        self.dilation = dilation
        self.embedding_hook = embedding_hook

        if verbose:
            print("\nEffNet Parameters:")
            print(f"{self.input_channels=}")
            print(f"{self.channels=}")
            print(f"{self.output_nerons=}")
            print(f"{self.depth=}")
            print(f"{self.expansion=}")
            print(f"{self.stride=}")
            print(f"{self.dilation=}")
            print(f"{self.embedding_hook=}")
            print("\n")

        self.stage1 = nn.Conv1d(
            self.input_channels,
            self.channels[0],
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=dilation,
        )  # 1 conv

        self.b0 = nn.BatchNorm1d(self.channels[0])

        self.stage2 = MBConv(
            self.channels[0], self.channels[1], self.expansion, self.depth[0], stride=2
        )

        self.stage3 = MBConv(
            self.channels[1], self.channels[2], self.expansion, self.depth[1], stride=2
        )

        self.Pool = nn.MaxPool1d(3, stride=1, padding=1)

        self.stage4 = MBConv(
            self.channels[2], self.channels[3], self.expansion, self.depth[2], stride=2
        )

        self.stage5 = MBConv(
            self.channels[3], self.channels[4], self.expansion, self.depth[3], stride=2
        )

        self.stage6 = MBConv(
            self.channels[4], self.channels[5], self.expansion, self.depth[4], stride=2
        )

        self.stage7 = MBConv(
            self.channels[5], self.channels[6], self.expansion, self.depth[5], stride=2
        )

        self.stage8 = MBConv(
            self.channels[6], self.channels[7], self.expansion, self.depth[6], stride=2
        )

        self.stage9 = nn.Conv1d(self.channels[7], self.channels[8], kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.num_extra_inputs = num_extra_inputs
        self.fc = nn.Linear(self.channels[8] + num_extra_inputs, self.output_nerons)
        self.fc.bias.data[0] = 0.275

    def forward(self, x: Tensor) -> Tensor:
        if self.num_extra_inputs > 0:
            x, extra_inputs = x

        x = self.b0(self.stage1(x))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.Pool(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.Pool(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.act(self.AAP(x)[:, :, 0])
        x = self.drop(x)

        if self.num_extra_inputs > 0:
            x = torch.cat((x, extra_inputs), 1)

        if self.embedding_hook:
            return x
        else:
            x = self.fc(x)
            return x

class MBConv(nn.Module):
    def __init__(
        self, in_channel, out_channels, expansion, layers, activation=nn.ReLU6, stride=2
    ):
        super().__init__()

        self.stack = OrderedDict()
        for i in range(0, layers - 1):
            self.stack["s" + str(i)] = Bottleneck(
                in_channel, in_channel, expansion, activation
            )

        self.stack["s" + str(layers + 1)] = Bottleneck(
            in_channel, out_channels, expansion, activation, stride=stride
        )

        self.stack = nn.Sequential(self.stack)

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stack(x)
        return self.bn(x)

class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        expansion: int,
        activation: Callable,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.stride = stride
        self.conv1 = nn.Conv1d(in_channel, in_channel * expansion, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channel * expansion,
            in_channel * expansion,
            kernel_size=3,
            groups=in_channel * expansion,
            padding=padding,
            stride=stride,
        )
        self.conv3 = nn.Conv1d(
            in_channel * expansion, out_channel, kernel_size=1, stride=1
        )
        self.b0 = nn.BatchNorm1d(in_channel * expansion)
        self.b1 = nn.BatchNorm1d(in_channel * expansion)
        self.d = nn.Dropout()
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            y = self.d(y)
            y = x + y
            return y
        else:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            return y

class ECGModel(pl.LightningModule):
    def __init__(
        self,
        model,
        index_labels=None,
        save_predictions_path=None,
    ):
        super().__init__()
        self.m = model
        self.labels = index_labels
        if self.labels is not None and isinstance(self.labels, str):
            self.labels = [self.labels]
        if isinstance(save_predictions_path, str):
            save_predictions_path = Path(save_predictions_path)
        self.save_predictions_path = save_predictions_path

    def prepare_batch(self, batch):
        if "labels" in batch and len(batch["labels"].shape) == 1:
            batch["labels"] = batch["labels"][:, None]
        return batch

    def forward(self, x):
        return self.m(x)

    def step(self, batch):
        batch = self.prepare_batch(batch)
        if "extra_inputs" not in batch.keys():
            y_pred = self(batch["primary_input"])
        else:
            x = (batch["primary_input"], batch["extra_inputs"])
            y_pred = self(x)

        return y_pred

    def predict_step(self, batch, batch_index):
        y_pred = self.step(batch)
        return {"filename": batch["filename"], "prediction": y_pred.cpu().numpy()}

    def on_predict_epoch_end(self, results):

        for i, predict_results in enumerate(results):
            filename_df = pd.DataFrame(
                {
                    "filename": np.concatenate(
                        [batch["filename"] for batch in predict_results]
                    )
                }
            )

            if self.labels is not None:
                columns = [f"{class_name}_preds" for class_name in self.labels]
            else:
                columns = ["preds"]
            outputs_df = pd.DataFrame(
                np.concatenate(
                    [batch["prediction"] for batch in predict_results], axis=0
                ),
                columns=columns,
            )

            prediction_df = pd.concat([filename_df, outputs_df], axis=1)

            dataloader = self.trainer.predict_dataloaders[i]
            manifest = dataloader.dataset.manifest
            prediction_df = prediction_df.merge(manifest, on="filename", how="outer")
            
            if self.save_predictions_path is not None:

                if ".csv" in self.save_predictions_path.name:
                    prediction_df.to_csv(
                        self.save_predictions_path.parent
                        / self.save_predictions_path.name.replace(".csv", f"_{i}_.csv"),
                        index=False,
                    )
                else:
                    prediction_df.to_csv(
                        self.save_predictions_path / f"dataloader_{i}_predictions.csv",
                        index=False,
                    )

class ECGDataset(Dataset):
    def __init__(
        self,
        data_path: Union[Path, str],
        manifest_path: Union[Path, str] = None,
        split: str = None,
        labels: List[str] = None,
        subsample: Union[int, float] = None,
        verbose: bool = True,
        verify_existing: bool = True,
        drop_na_labels: bool = True,
        first_lead_only = False
    ):
        
        self.verbose = verbose
        self.data_path = Path(data_path)
        self.split = split
        self.subsample = subsample
        self.verify_existing = verify_existing
        self.drop_na_labels = drop_na_labels
        self.first_lead_only = first_lead_only

        

        self.labels = labels
        if self.labels is None and self.verbose:
            print(
                "No label column names were provided, only filenames and inputs will be returned."
            )
        if (self.labels is not None) and isinstance(self.labels, str):
            self.labels = [self.labels]

        # Read manifest file
        self.manifest_path = Path(manifest_path)

        if self.manifest_path.exists():
            self.manifest = pd.read_csv(self.manifest_path, low_memory=False)
        else:
            self.manifest = pd.DataFrame(
                {
                    "filename": os.listdir(self.data_path),
                }
            )

        # Usually set to "train", "val", or "test". If set to None, the entire manifest is used.
        if self.split is not None:
            self.manifest = self.manifest[self.manifest["split"] == self.split]
        if self.verbose:
            print(
                f"Manifest loaded. \nSplit: {self.split}\nLength: {len(self.manifest):,}"
            )

        # Make sure all files actually exist. This can be disabled for efficiency if
        # you have an especially large dataset
        if self.verify_existing:
            old_len = len(self.manifest)
            existing_files = os.listdir(self.data_path)
            self.manifest = self.manifest[
                self.manifest["filename"].isin(existing_files)
            ]
            new_len = len(self.manifest)
            if self.verbose:
                print(
                    f"{old_len - new_len} files in the manifest are missing from {self.data_path}."
                )
        elif (not self.verify_existing) and self.verbose:
            print(
                f"self.verify_existing is set to False, so it's possible for the manifest to contain filenames which are not present in {data_path}"
            )

        # Option to subsample dataset for doing smaller, faster runs
        if self.subsample is not None:
            if isinstance(self.subsample, int):
                self.manifest = self.manifest.sample(n=self.subsample)
            else:
                self.manifest = self.manifest.sample(frac=self.subsample)
            if verbose:
                print(f"{self.subsample} examples subsampled.")

        # Make sure that there are no NAN labels
        if (self.labels is not None) and self.drop_na_labels:
            old_len = len(self.manifest)
            self.manifest = self.manifest.dropna(subset=self.labels)
            new_len = len(self.manifest)
            if self.verbose:
                print(
                    f"{old_len - new_len} examples contained NaN value(s) in their labels and were dropped."
                )
        elif (self.labels is not None) and (not self.drop_na_labels):
            print(
                "self.drop_na_labels is set to False, so it's possible for the manifest to contain NaN-valued labels."
            )


    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        output = {}
        row = self.manifest.iloc[index]
        filename = row["filename"]
        output["filename"] = filename

        # self.read_file expected in child classes
        primary_input = self.read_file(self.data_path / filename, row)

        labels = row[self.labels] if self.labels is not None else None
        if self.labels is not None and not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.float32)

        if not torch.is_tensor(primary_input):
            primary_input = torch.tensor(primary_input, dtype=torch.float32)
        output["primary_input"] = primary_input

        output["labels"] = labels

        return output
    def read_file(self, filepath, row=None):
        # ECGs are usually stored as .npy files.
        file = np.load(filepath)
        if file.shape[0] != 12:
            file = file.T
        file = torch.tensor(file).float()

        if self.first_lead_only:
            # accessing with a slize [0:1] is deliberate, it makes the final shape (1, N) instead of just (N)
            file = file[0:1]

        # Final shape should ideally be NumLeadsxTime(or NumLeadsxTime depending on the resolution of the ECG)
        return file
