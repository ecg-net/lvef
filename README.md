# LVEF_ekg_external_validation

Classification and Regression Models for using ECGs and Echo Pairs obtained within 31 days of one another.

# Directions For Running Inference

Navigate to the direcfory of the repository at the command line. 

# Running Inference

Run the following in the command line, specifying arguments: 

python predict.py --task <task> --label_col <label_col> --split < split> --manifest_path <manifest_path> --data_path <data_path> --save_predictions_path <save_predictions_path>

--label_col: specify the ground truth label column of your dataset (code will then remove EKGs with NaN ground truth labels) </br>
--split: specify train, val, or test (for the cases in your manifest). If you want to use all the specify None </br>
--manifest_path: Path to your manifest </br>
--data_path: Path to Normalized NPYs </br>
--save_predictions_path: Path where a predictions csv with model predictions will be saved (Defaults to current directory if left blank) </br>

For EF Regression Model: Final predictions will be in a column called "EF_preds" in the regression output
For EF Classification Model: Final predictions will be in a column called "preds" in the regression output</br>

Your manifest must have a "filename" column with names of the NPYs you are running inference on.</br>
Your manifest may have a split column. You can specify the split on which you want to run inference in the split argument. Otherwise, if your manifest doesn't have a split column and/or you want to run inference on all NPYs in your manifest, specify None in the split argument.


# Example Code: 

python predict.py --label_col EF_2D --split test --manifest_path /workspace/Amey/wandb_runs_and_manifests/wandb_runs/lvef_ekg_wandb/lvef_31_days_stride8_dilation2/data/ekg_w_ef_31_days.csv --data_path /workspace/data/drives/sdc/Amey/ekgs/ecg_npy_denoise_wbr_norm_2023_01_13

### Obtaining Stats and Figures</br>

After running predict.py, run all cells in the notebook "analyze_predictions.ipynb"

This will output bootstrapped statistics for each model and a calibration plot comparing Regression and Classification Model results on your dataset.

<img width="1083" alt="image" src="https://github.com/echonet/MR/assets/111397367/03b51ec7-f062-4c7d-8657-3be85d49128c">

</br></br> </br>

![](https://github.com/echonet/MR/blob/master/analyze_predictions.gif?raw=true)







