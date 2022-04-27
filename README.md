# CS598_FINAL_PROJECT
_Deep Learning for Healthcare - Final Project - Classification of ICD-9 codes to Clinical text_

###**SYSTEM REQUIREMENTS:**

- Python 3.10 - [Download Python](https://www.python.org/)
- fastText version 0.9.2+
- numpy - version 1.22.3+
- pandas - version 1.4.2+
- torch - version 1.11.0+
- gensim - version 4.1.2+
- Packages in interpreter used for development at shown below:
  - ![img.png](img.png)


###**DATA ACCESS HOW TO:**

- Data can be downloaded from the MIMIC-III Website at: [Physionet](https://physionet.org/content/mimiciii/1.4/)
  - Use version 1.4 (in link above) - Access will need to be requested
  - Files needed:
    - D_ICD_DIAGNOSES.csv.gz
    - D_ICD_PROCEDURES.csv.gz
    - DIAGNOSES_ICD.csv.gz
      - Contains Diagnosis codes (ICD9) for patients from NOTEEVENTS.csv.gz
    - NOTEVENTS.csv.gz
      - Contains clinical notes for patients

###**DESCRIPTION OF PROJECT:**

- clean_data.py
  - Used for data processing of datasets from MIMIC-III
- train_model.py
  - Used for training the fastText model
- fasttext-0.9.2-cp310-cp310-win_amd64.whl
  - Some users may encounter issues when trying to install fastText (used for binary classification) to their local environments. This whl file can be used to manually install fastText to your venv
    - Command: **pip install fasttext-0.9.2-cp310-cp310-win_amd64.whl**

###**HOW TO RUN**
    1. Clean Data
        - Run clean_data.py from venv, or from terminal with "python clean_data.py"
        - You will see cleansed data within the cleansed_data directory

    2. Train Model for bag of tricks (NOT NEEDED UNLESS YOU WANT TO RETRAIN!)
        - To retrain, run train_models_bot.py from venv, or from terminal with "python train_models_bot.py"
            - Model for regular ICD codes is saved at ./models/model_reg.bin
            - Model for rolled ICD codes is saved at ./models/model_reg.bin

    3. Train Model for CNN (NOT NEEDED UNLESS YOU WANT TO RETRAIN!)
        - To retrain, run train_models_cnn.py from venc, or from terminal with "python train_models_cnn.py"

    4. Evaluate model performance on pre-processed data
        - Run evaluate models.py from venv, or from terminal with "python evaluate_models.py"
