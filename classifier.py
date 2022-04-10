# CTOOMBS 11/27/2019
#   This is the final project for CS410 Text Information systems
#   This program will read in tweets and classify them with a label of either SARCASM or NOT_SARCASM
#   The output of running this program will be a file called answer.txt
import logging
import os
import re
import json
import csv
import numpy as np

# Genism Libraries, used for vector space modeling
import gensim
import pandas as pd
from gensim.models import word2vec

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
    proc_filepath = './data/' + proc_file
    cleaned_filepath = './cleansed_data/' + diag_file

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
    diagnoses_df = diagnoses_df[diagnoses_df['SUBJECT_ID'].isin(unique_subjects_list)].drop(columns=['ROW_ID', 'SEQ_NUM'])
    diagnoses_df['ICD9_CODE_ROLLED'] = diagnoses_df['ICD9_CODE'].apply(lambda x: str(x)[0:3])

    my_logger.info('  FILTERED DIAG FILE LENGTH: ' + str(len(diagnoses_df)))
    my_logger.info('  NUMBER OF UNIQUE CODES: ' + str(diagnoses_df.ICD9_CODE.nunique()))
    my_logger.info('  NUMBER OF UNIQUE ROLLED CODES: ' + str(diagnoses_df.ICD9_CODE_ROLLED.nunique()))

    my_logger.info(diagnoses_df)

    my_logger.info('DIAG File Cleansed...')

    return unique_subjects_list, unique_admits_list, diagnoses_df


# This method will take in a filename and populate a list of dictionaries with tweets
def cleanse_notes(unique_subjects: list, unique_admits: list):
    noteevents_filepath = './data/' + noteevents_file
    cleaned_filepath = './cleansed_data/' + noteevents_file

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
    with pd.read_csv(noteevents_filepath, compression='gzip', header=0, sep=',', quotechar='"', chunksize=5000) as reader:
        for chunk in reader:
            chunk_num = chunk_num + 1
            my_logger.info('  PROCESSING CHUNK: ' + str(chunk_num))

            # Only keep reports if subject is in unique subjects and the HADM_ID is in the cleansed diagnosis dataframe
            temp = chunk[(chunk['SUBJECT_ID'].isin(unique_subjects)) & (chunk['HADM_ID'].isin(unique_admits))].drop(columns=['CGID', 'CHARTDATE', 'CHARTTIME', 'DESCRIPTION', 'ISERROR', 'ROW_ID', 'STORETIME'])

            # PER PAPER CLEANSE THE LINES
            # Remove punctuation to whitespaces except for apostrophe
            temp['TEXT'] = temp['TEXT'].apply(lambda x: re.sub(r"[,.:;@#?!&$*\[\]]+\ *", " ", x))

            # Digits replaced by letter 'd'
            temp['TEXT'] = temp['TEXT'].apply(lambda x: re.sub(r"[0-9]", "d", x))

            # Convert all characters to lowercase
            temp['TEXT'] = temp['TEXT'].apply(lambda x: x.lower())

            # Remove excess whitespace
            temp['TEXT'] = temp['TEXT'].apply(lambda x: ' '.join(x.split()))

            notes_list.append(temp)
            #my_logger.info(temp)
            #raise

    # convert notes_list to consolidate DF
    notes_df = pd.concat(notes_list)

    my_logger.info(" LENGTH OF NOTES DF: " + str(len(notes_df)))

    return notes_df

# Main method - python will automatically run this
if __name__ == '__main__':
    unique_subject_list, unique_admit_list, diagnoses_df = cleanse_diagnoses()
    notes_df = cleanse_notes(unique_subject_list, unique_admit_list)