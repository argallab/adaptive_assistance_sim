# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import os
import pickle
import collections
import argparse
import random

def chunks(l, n):
    '''
    code taken from https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/
    '''
    for i in range(0, len(l), n):
        yield l[i:i+n]

def create_experiment_blocks(args):
    metadata_dir = args.metadata_dir
    metadata_filename = 'assistance_to_pkl_index.pkl'
    subject_id = args.subject_id
    num_blocks = args.num_blocks

    assert os.path.exists(os.path.join(metadata_dir, metadata_filename))
    with open(os.path.join(metadata_dir, metadata_filename), 'rb') as fp:
        assistance_to_pkl_index = pickle.load(fp)

    total_num_trials = args.total_num_trials #TODO remove hardcoding
    assert len(assistance_to_pkl_index[0]) == len(assistance_to_pkl_index[1]) == len(assistance_to_pkl_index[2])
    num_trials_per_assistance = len(assistance_to_pkl_index[0])
    num_trials_per_block = total_num_trials/num_blocks

    #assistance type 0 - Filter, 1 -Corrective, 2 - No Assistance
    filter_assistance_list = assistance_to_pkl_index[0]
    random.shuffle(filter_assistance_list)
    filter_assistance_blocks = list(chunks(filter_assistance_list, num_trials_per_block))
    for i, filter_assistance_block in enumerate(filter_assistance_blocks):
        filename = subject_id + '_filter_assistance_' + str(i) + '_num_blocks_'+str(num_blocks)+'.pkl'
        with open(os.path.join(metadata_dir, filename), 'wb') as fp:
            pickle.dump(filter_assistance_block, fp)

    corrective_assistance_list = assistance_to_pkl_index[1]
    random.shuffle(corrective_assistance_list)
    corrective_assistance_blocks = list(chunks(corrective_assistance_list, num_trials_per_block))
    for i, corrective_assistance_block in enumerate(corrective_assistance_blocks):
        filename = subject_id + '_corrective_assistance_' + str(i) + '_num_blocks_'+str(num_blocks)+'.pkl'
        with open(os.path.join(metadata_dir, filename), 'wb') as fp:
            pickle.dump(corrective_assistance_block, fp)

    no_assistance_list = assistance_to_pkl_index[2]
    random.shuffle(no_assistance_list)
    no_assistance_blocks = list(chunks(no_assistance_list, num_trials_per_block))
    for i, no_assistance_block in enumerate(no_assistance_blocks):
        filename = subject_id + '_no_assistance_' + str(i) + '_num_blocks_'+str(num_blocks)+'.pkl'
        with open(os.path.join(metadata_dir, filename), 'wb') as fp:
            pickle.dump(no_assistance_block, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_dir', dest='metadata_dir',default=os.path.join(os.getcwd(), 'metadata_dir'), help="The directory where metadata of trials will be stored")
    parser.add_argument('--subject_id', dest='subject_id',default='deepak', help="unique_identifier for subject")
    parser.add_argument('--num_blocks', dest='num_blocks', default=6, help='total number of blocks')
    parser.add_argument('--total_num_trials', dest='total_num_trials', default=36, help='total number of trials')

    args = parser.parse_args()
    create_experiment_blocks(args)
