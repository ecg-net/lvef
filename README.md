# lvef_and_amyloid_ekg_external_validation

# Directions For Running Inference

Navigate to the direcfory of the repository at the command line

# Running Inference

Run the following in the command line, specifying arguments: 

python predict.py --task <task> --label_col <label_col> --split < split> --manifest_path <manifest_path> --data_path <data_path> --save_predictions_path <save_predictions_path>

--task: specify EF or amyloid </br>
--label_col: specify the ground truth label column of your dataset (code will then remove EKGs with NaN ground truth labels) </br>
--split: specify train, val, or test (for the cases in your manifest). If you want to use all the specify None </br>
--manifest_path: Path to your manifest </br>
--data_path: Path to Normalized NPYs </br>
--save_predictions_path: Path where a predictions csv with model predictions will be saved (Defaults to current directory if left blank) </br>

For EF Task: Final predictions will be in a column called "EF_preds"</br>
For Amyloid Task: Final Predictions will be in a column called "preds"</br>

Your manifest must have a "filename" column with names of the NPYs you are running inference on.</br>
Your manifest may have a split column. You can specify the split on which you want to run inference in the split argument. Otherwise, if your manifest doesn't have a split column and/or you want to run inference on all NPYs in your manifest, specify None in the split argument.


# Example Code: 
python predict.py --task EF --label_col EF_2D --split test --manifest_path /workspace/Amey/lvef_ekg_external_validation/ef_manifest_no_split.csv --data_path /workspace/data/drives/sdc/Amey/ekgs/ecg_npy_denoise_wbr_norm_2023_01_13

### Obtaining AUC
For the amyloid model, the code will automatically yield AUC and display it in the command line. </br>

EF predictions will also be normalized EF values as the model is actually a regression model. The prediction code will automatically convert them from normalized EF predictions to EF values (using our mean and standard deviation). To obtain AUC, binarize the EF ground truth column based on a threshold of choice and calculate AUC based on the binarized labels and predicted EF values.
