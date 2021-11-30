#!/bin/bash

# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

# extract topics from bag files
# generate p(Um|Ui)
# generate p(Um|a)
# generate p(Ui|a)
# generate p(Ui|a) optimization
# plot distributions


# input: bagfile name, subject name
# bagfile=$1
# echo "Bagfile: $bagfile"
subject_id=$1
echo "Subejct id: $subject_id"

search_dir="/root/.ros/aa/$subject_id/"
echo "$search_dir"

for full_file in ${search_dir}*.bag;
do
	file_name=${full_file##*/}
	name="$(cut -d'_' -f1 <<<$file_name)"
	condition="$(cut -d'_' -f2 <<<$file_name)"
	# echo "$file_name"
	# echo "$name"
	# echo "$condition"
	if [[ "$name" == "$subject_id" ]]; then
		if [[ "$condition" == 'disamb' ]] || [[ "$condition" == 'control' ]]; then # p(Um|Ui)
			trial_bag=$full_file
			block="$(cut -d'_' -f4<<<$file_name)"
			echo $trial_bag
			echo $block
			# extract data
			trial_block_name="${subject_id}_${condition}_condition_${block}"
			echo "$trial_block_name"
			echo "Extracting: $trial_bag"
			python extract_topics_from_bag.py $trial_bag "$trial_block_name"

			# # Build main study dataframes:
			# echo "Concatinating data from $trial_block_name and creating trial data frames"
			# python main_study_concatenate_topics_per_trial.py -b ${trial_block_name}
		fi
	fi
done
