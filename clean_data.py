# CTOOMBS 04/26/2022
#   This script will cleanse input data for notes and diagnoses for the labeling task
import logging
import os
import re

# Genism Libraries, used for vector space modeling
import numpy as np
import pandas as pd

# Set Logging -- basic configuration
logging.basicConfig(format='%(asctime)s -- %(levelname)s: %(message)s',
                    level=logging.NOTSET,
                    datefmt='%Y-%m-%d %H:%M:%S')
my_logger = logging.getLogger('classifier_project')

# File names for the diagnoses and note events file. This can be changed if you get from another source
diag_file = 'DIAGNOSES_ICD.csv.gz'
proc_file = 'D_ICD_PROCEDURES.csv.gz'
noteevents_file = 'NOTEEVENTS.csv.gz'


# This method will take in a filename and populate a list of dictionaries with tweets
def cleanse_diagnoses():
    diag_filepath = './data/' + diag_file

    if os.path.exists(diag_filepath):
        my_logger.info('DIAG File Exists!')
    else:
        my_logger.error('DIAG File DOES NOT EXIST! Input files should be contained in the data directory')
        raise

    # ROW_ID/SUBJECT_ID/HADM_ID/SEQ_NUM/ICD9_CODE
    my_logger.info('Cleansing DIAG file -- ' + diag_file)

    diagnoses_df = pd.read_csv(diag_filepath, compression='gzip',
                               header=0, sep=',', quotechar='"')

    my_logger.info('  DIAG FILE LENGTH: ' + str(len(diagnoses_df)))

    # Filter out rows that are NOT diabetic - 250 is ICD code for DIABETES
    unique_subjects = diagnoses_df[diagnoses_df['ICD9_CODE'].str.startswith('250', na=False)]

    # Get the list of unique patients and admits - This will be used by the notes filtering!
    unique_subjects_list = unique_subjects['SUBJECT_ID'].unique()
    unique_admits_list = unique_subjects['HADM_ID'].unique()

    my_logger.info('  NUMBER OF UNIQUE PATIENTS: ' + str(len(unique_subjects_list)))
    my_logger.info('  NUMBER OF UNIQUE ADMITS: ' + str(len(unique_admits_list)))

    # Filter out DIAG Dataframe to remove subjects which are not in the unique subject list
    # Remove ROW_ID, SEQ_NUM columns as we don't need these and this takes up memory - keeping HADM_ID as it keeps
    # Record of hospital admit
    diagnoses_df = diagnoses_df[diagnoses_df['SUBJECT_ID'].isin(unique_subjects_list)].drop(
        columns=['ROW_ID', 'SEQ_NUM'])
    diagnoses_df['ICD9_CODE_ROLLED'] = diagnoses_df['ICD9_CODE'].apply(lambda x: str(x)[0:3])

    my_logger.info('  FILTERED DIAG FILE LENGTH: ' + str(len(diagnoses_df)))
    my_logger.info('  NUMBER OF UNIQUE CODES: ' + str(diagnoses_df.ICD9_CODE.nunique()))
    my_logger.info('  NUMBER OF UNIQUE ROLLED CODES: ' + str(diagnoses_df.ICD9_CODE_ROLLED.nunique()))

    my_logger.info('DIAG File Cleansed...')

    return unique_subjects_list, unique_admits_list, diagnoses_df


# This method will take in a filename and populate a list of dictionaries with tweets
def cleanse_notes(unique_subjects: list, unique_admits: list):
    noteevents_filepath = './data/' + noteevents_file

    if os.path.exists(noteevents_filepath):
        my_logger.info('NOTES File Exists!')
    else:
        my_logger.error('NOTES file DOES NOT EXIST! Input files should be contained in the data directory')
        raise

    # ['CATEGORY', 'CGID', 'CHARTDATE', 'CHARTTIME', 'DESCRIPTION', 'HADM_ID', 'ISERROR', 'ROW_ID', 'STORETIME', 'SUBJECT_ID', 'TEXT']
    my_logger.info('Cleansing NOTES file -- ' + noteevents_file)

    chunk_num = 0
    notes_list = []

    # Requires pandas 1.25>
    with pd.read_csv(noteevents_filepath, compression='gzip', header=0, sep=',', quotechar='"',
                     chunksize=5000) as reader:
        for chunk in reader:
            chunk_num = chunk_num + 1
            my_logger.info('  PROCESSING CHUNK: ' + str(chunk_num))

            # Only keep reports if subject is in unique subjects and the HADM_ID is in the cleansed diagnosis dataframe
            # ADDED: Only keep discharge summary or output file is far too large!!!
            temp = chunk[(chunk['SUBJECT_ID'].isin(unique_subjects)) & (chunk['HADM_ID'].isin(unique_admits)) & (chunk['CATEGORY'].isin(['Discharge summary']))].drop(
                columns=['CGID', 'CHARTDATE', 'CHARTTIME', 'DESCRIPTION', 'ISERROR', 'ROW_ID', 'STORETIME'])

            # PER PAPER CLEANSE THE LINES
            # Remove punctuation to whitespaces except for apostrophe
            temp['TEXT'] = temp['TEXT'].apply(lambda x: re.sub(r"[,.:;_@#?!&$*\[\]]+\ *", " ", x))

            # Digits replaced by letter 'd'
            temp['TEXT'] = temp['TEXT'].apply(lambda x: re.sub(r"[0-9]", "d", x))

            # Convert all characters to lowercase
            temp['TEXT'] = temp['TEXT'].apply(lambda x: x.lower())

            # Remove excess whitespace
            temp['TEXT'] = temp['TEXT'].apply(lambda x: ' '.join(x.split()))

            notes_list.append(temp)
            # my_logger.info(temp)
            # raise

    # convert notes_list to consolidate DF
    notes_df = pd.concat(notes_list)

    my_logger.info(" LENGTH OF NOTES DF: " + str(len(notes_df)))

    return notes_df


# This method will save off the notes / diagnoses in a cleansed format so we will not need to keep reading each pass
def save_training_files(notes_df, diagnoses_df, unique_admit_list):
    cleansed_input_filepath = './cleansed_data/'
    cleansed_regular = cleansed_input_filepath + 'regular_input.txt'
    cleansed_rolled = cleansed_input_filepath + 'rolled_input.txt'

    # We will save off two versions of this cleansed file, the rolled and the unrolled ICD 9 codes
    # For each hospital visit, output the lines for REGULAR ICD9 codes
    with open(cleansed_regular, 'w', encoding='utf8') as f:
        for item in np.nditer(unique_admit_list):

            diagnosis = diagnoses_df[diagnoses_df['HADM_ID'] == item]
            note = notes_df[notes_df['HADM_ID'] == item]

            # For each diagnosis per hospital admit,
            for row in diagnosis.itertuples(index=True):
                if row.Index == 1:
                    f.write('__label__' + str(row.ICD9_CODE))
                else:
                    f.write(' __label__' + str(row.ICD9_CODE))

            # Add the note to the end of the line
            for row in note.itertuples(index=True):
                f.write(' ' + str(row.TEXT))

            # Next Line
            f.write('\n')

    # For each hospital visit, output lines for ROLLED ICD9 codes
    with open(cleansed_rolled, 'w', encoding='utf8') as f:
        for item in np.nditer(unique_admit_list):

            diagnosis = diagnoses_df[diagnoses_df['HADM_ID'] == item]
            note = notes_df[notes_df['HADM_ID'] == item]

            # For each diagnosis per hospital admit,
            for row in diagnosis.itertuples(index=True):
                if row.Index == 1:
                    f.write('__label__' + str(row.ICD9_CODE_ROLLED))
                else:
                    f.write(' __label__' + str(row.ICD9_CODE_ROLLED))

            # Add the note to the end of the line
            for row in note.itertuples(index=True):
                f.write(' ' + str(row.TEXT))

            # Next Line
            f.write('\n')



# Main method - python will automatically run this
if __name__ == '__main__':
    unique_subject_list, unique_admit_list, diagnoses_df = cleanse_diagnoses()
    notes_df = cleanse_notes(unique_subject_list, unique_admit_list)

    # Debug only
    #my_logger.info(notes_df['CATEGORY'].unique())
    #my_logger.info(notes_df[notes_df['HADM_ID'] == 140784])
    #my_logger.info(diagnoses_df[diagnoses_df['HADM_ID'] == 140784])
    save_training_files(notes_df, diagnoses_df, unique_admit_list)
    my_logger.info("CLEANSING COMPLETE")
