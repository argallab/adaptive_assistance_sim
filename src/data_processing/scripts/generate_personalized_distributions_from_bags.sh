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

search_dir="/root/.ros/"

declare -A p_ui_given_a_bags_array
i=0
for full_file in ${search_dir}*.bag;
do
	file_name=${full_file##*/}
	name="$(cut -d'_' -f1 <<<$file_name)"
	p_of="$(cut -d'_' -f3 <<<$file_name)"
	given="$(cut -d'_' -f5 <<<$file_name)"
	# echo "$file_name"
	# echo "$name"
	if [[ "$name" == "$subject_id" ]]; then
		if [[ "$p_of" == 'um' ]] && [[ "$given" == 'ui' ]]; then # p(Um|Ui)
			p_um_given_ui_bag=$full_file
		fi
		if [[ "$p_of" == 'um' ]] && [[ "$given" == 'a' ]]; then # p(Um|a)
			p_um_given_a_bag=$full_file
		fi
		if [[ "$p_of" == 'ui' ]] && [[ "$given" == 'a' ]]; then # p(Ui|a)
			# p_ui_given_a_bag=$full_file
			p_ui_given_a_bags_array[$i]=$full_file
			i=i+1
		fi
		# echo $full_file
	fi
done


# get latest p_ui_given_a bag
max_y=0
max_d=0
max_month=0
max_h=0
max_m=0
max_x=0
i=0
for file in "${p_ui_given_a_bags_array[@]}";
do
	file_name=${file##*/}
	datetime="$(cut -d'_' -f6 <<<$file_name)"

	year="$(cut -d'-' -f1 <<<$datetime)"
	month="$(cut -d'-' -f2 <<<$datetime)"
	day="$(cut -d'-' -f3 <<<$datetime)"
	hour="$(cut -d'-' -f4 <<<$datetime)"
	mins="$(cut -d'-' -f5 <<<$datetime)"
	secs="$(cut -d'-' -f6 <<<$datetime)"
	secs="$(cut -d'.' -f1 <<<$secs)"

	if (( $year == $max_y )); then
		if (( $month == $max_month )); then
			if (( $day == $max_d )); then
				if (( $hour == $max_h )); then
					if (( $mins == $max_m)); then
						if (( $secs > $max_s )); then
							max_s=$secs
							i=$i
						fi
					fi
					if (( $mins > $max_m )); then
						max_m=$mins
						i=$i
					fi
				fi
				if (( $hour > $max_h )); then
					max_h=$hour
					i=$i
				fi
			fi
			if (( $day > $max_d )); then
				max_d=$day
				i=$i
			fi
		fi
		if (( $month > $max_month )); then
			max_month=$month
			i=$i
		fi
	fi
	if (( $year > $max_y )); then
		max_y=$year
		i=$i
	fi

	i=i+1
done

p_ui_given_a_bag=${p_ui_given_a_bags_array[$i]}
if [[ "${#p_ui_given_a_bags_array[@]}" == 1 ]]; then # if lenght of array is one, return just that one file (otherwise gives empty)
	p_ui_given_a_bag=$p_ui_given_a_bags_array
fi

# p(Um|Ui): (interface_distorition)
echo "Extracting: $p_um_given_ui_bag"
python extract_topics_from_bag.py $p_um_given_ui_bag "${subject_id}_p_um_given_ui"

# p(Um|a): (command_following)
echo "Extracting: $p_um_given_a_bag"
python extract_topics_from_bag.py $p_um_given_a_bag "${subject_id}_p_um_given_a"

# P(Ui|a) (internal_model)
echo "Extracting: $p_ui_given_a_bag"
python extract_topics_from_bag.py $p_ui_given_a_bag "${subject_id}_p_ui_given_a"


# Build distributions:

# P(Um|Ui)
echo "Generating p(um|ui)"
python p_um_given_ui_distribution_preprocessing.py -id ${subject_id}

# P(Um|a)
echo "Generating p(um|a)"
python p_um_given_a_distribution_preprocessing.py -id ${subject_id}

# P(Ui|a)
echo "Generating p(ui|a)"
python p_ui_given_a_distribution_preprocessing.py -id ${subject_id}

# P(Um|a) optimization
echo "Generating p(ui|a) implicity using optimization"
python p_ui_given_a_distribution_preprocessing_implicit.py -id ${subject_id}
