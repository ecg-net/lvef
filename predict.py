import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import os
from utils import ECGDataset, EffNet, ECGModel, sigmoid
import pandas as pd
from sklearn import metrics
import argparse


if __name__ == '__main__':
    if os.name == 'nt':
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        print('Windows detected!')

parser = argparse.ArgumentParser(description='Run inference on a trained model: either LVEF or amyloid')
parser.add_argument('--label_col', type=str, help='Name of ground truth label column in manifest; drops cases with NaN ground truth')
parser.add_argument('--split', type=str, choices = ['train','val','test','None'], help = 'Split to run inference on. None, will run inference on all splits')
parser.add_argument('--manifest_path', type=str, help='Path to Manifest File')
parser.add_argument('--data_path', type=str, help='Path to Normalized EKGs')
parser.add_argument('--save_predictions_path', type=str, default='./')
args = parser.parse_args()


if args.split == 'None':
    args.split = None

data_path = args.data_path
manifest_path = args.manifest_path

test_ds = ECGDataset(split=args.split,
                    data_path=data_path, manifest_path=manifest_path, labels=[args.label_col])

test_dl = DataLoader(test_ds, num_workers=24, batch_size=500, drop_last=False, shuffle=False)
backbone = EffNet(output_neurons=1)
model = ECGModel(backbone, save_predictions_path=args.save_predictions_path)

weights_path = 'lvef_regression.pt'
print(model.load_state_dict(torch.load(weights_path)))
trainer = Trainer(gpus=0)
trainer.predict(model, dataloaders=test_dl)
os.rename('dataloader_0_predictions.csv', 'EF_predictions_regression.csv')
predictions = pd.read_csv('EF_predictions_regression.csv')
stdev = 16.283926568588136
mean = 55.55226219042883
predictions['EF_preds'] = (predictions['preds']*stdev) + mean
predictions = predictions.rename(cols = {str(args.label_col):'EF_2D'})
predictions.to_csv('EF_predictions_regression.csv', index = False)

weights_path = 'EF_binary_35.pt'
print(model.load_state_dict(torch.load(weights_path)))
trainer = Trainer(gpus=0)
trainer.predict(model, dataloaders=test_dl)
os.rename('dataloader_0_predictions.csv', 'EF_predictions_classification.csv')
predictions = pd.read_csv('EF_predictions_classification.csv')
predictions.preds = predictions.preds.apply(sigmoid)
predictions = predictions.rename(cols = {str(args.label_col):'EF_2D'})
predictions.to_csv('EF_predictions_classification.csv', index = False)