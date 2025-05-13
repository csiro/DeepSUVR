# Installation

First need to create a python virtual environment, and install the required packages (Note that this will install the cpu version of pytorch - you could install the GPU version, but it is fairly fast just on CPU, so GPU is not really needed)

```python -m venv venv
. venv/bin/activate

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```



To run DeepSUVR, all the images need to be spatially normalised using SPM.
The list of spatially normlaised images needs to be provided in a csv file with ID,TP,Tracer,Filename, with Filename being the spatially normalised PET image (the filename needs to start with w). 
We provide a subset of the GAAIN Calibration dataset in the Test/Calibration folder, with the corresponding csv file (Test/test_Calibration.csv) for testing.

DeepSUVR can then be run using

```
python DeepSUVR.py --in_csv Test/test_Calibration.csv --out_csv prediction.csv --checkpoints Models/*.tar
```

To run the prediction using DeepSUVR-derived reference and target masks, this can be run with

```
python DeepSUVR_Masks.py --in_csv Test/test_Calibration.csv --out_csv prediction_mask.csv
```
We also provide our prediction results for these 2 experiments in Test/prediction_ref.csv and Test/prediction_mask_ref.csv to be used to cross check that the same results are obtained.
