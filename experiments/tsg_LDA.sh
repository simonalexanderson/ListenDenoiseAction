#!/bin/bash

# # Dance
checkpoint=pretrained_models/tsg_LDA.ckpt
dest_dir=results/generated/tsg_LDA

if [ ! -d "${dest_dir}" ]; then
    mkdir -p "${dest_dir}"
fi

data_dir=data/tsg/
basenames=$(cat "${data_dir}/gen_files.txt")
start=0
seed=150
fps=30
trim_s=0
length_s=10
trim=$((trim_s*fps))
length=$((length_s*fps))
model_suffix="LDA"
fixed_seed=false
fast=false
gpu="cuda:0"
render_video=true

for filebase in $basenames;
do
	start=0
	for postfix in 0 1 2 3 4 5 6 7 8 9 10 11
	do			
		style=$(echo $filebase | awk -F "_" '{print $2}')
		input_file=${filebase}.audio29_${fps}fps.pkl		
		output_file=${filebase::-3}_${postfix}
		
		echo Generating motion from ${input_file} to ${dest_dir}/${output_file}		
		echo "start=${start}, len=${length}, postfix=${postfix}, seed=${seed}"

		python synthesize.py --checkpoints="${checkpoint}" --data_dirs="${data_dir}" --input_files="${input_file}" --start=${start} --end=${length} --trim=${trim} --seed=${seed} --postfix=${postfix} --dest_dir=${dest_dir} --gpu=${gpu} --outfile=${output_file} --video=${render_video}
		if [ "$fixed_seed" != "true" ]; then
			seed=$((seed+1))
		fi
		echo seed=$seed
		python utils/cut_wav.py ${data_dir}/${filebase::-3}.wav $(((start+trim)/fps)) $(((start+length-trim)/fps)) ${postfix} ${dest_dir}
		ffmpeg -y -i ${dest_dir}/${output_file}.mp4 -i ${dest_dir}/${filebase::-3}_${postfix}.wav ${dest_dir}/${output_file}_audio.mp4
		rm ${dest_dir}/${output_file}.mp4
		
		start=$((start+length))
	done
done
