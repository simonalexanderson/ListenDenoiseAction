#!/bin/bash

checkpoint=pretrained_models/zeggs_LDA.ckpt
dest_dir=results/generated/zeggs_LDA

if [ ! -d "${dest_dir}" ]; then
    mkdir -p "${dest_dir}"
fi

data_dir=data/zeggs
styles=("Happy" "Old" "Relaxed" "Angry" "Speech")

start=0
seed=150
fps=30
trim_s=0
length_s=10
trim=$((trim_s*fps))
length=$((length_s*fps))
fixed_seed=false
gpu="cuda:6"
render_video=true

for wavfile in "004_Neutral_3_x_1_0_00" "005_Neutral_4_x_1_0_00" 
do
	for style in ${styles[@]}; do
		start=0
		for postfix in 0 1 2 3 4 5 6 7 8 9 10 11
		do
			input_file=${src_dir}/${wavfile}.audio29_${fps}fps.pkl
			
			output_file=${wavfile::-3}_${postfix}_${style}
			
			echo "start=${start}, len=${length}, postfix=${postfix}, seed=${seed}"
			python synthesize.py --checkpoints="${checkpoint}" --data_dirs="${data_dir}" --input_files="${input_file}" --styles="${style}" --start=${start} --end=${length} --seed=${seed} --postfix=${postfix} --trim=${trim} --dest_dir=${dest_dir} --gpu=${gpu} --video=${render_video} --outfile=${output_file}
			if [ "$fixed_seed" != "true" ]; then
				seed=$((seed+1))
			fi 
			echo seed=$seed
			python utils/cut_wav.py ${data_dir}/${wavfile::-3}.wav $(((start+trim)/fps)) $(((start+length-trim)/fps)) ${postfix} ${dest_dir}
			if [ "$render_video" == "true" ]; then
				ffmpeg -y -i ${dest_dir}/${output_file}.mp4 -i ${dest_dir}/${wavfile::-3}_${postfix}.wav ${dest_dir}/${output_file}_audio.mp4
				rm ${dest_dir}/${output_file}.mp4
			fi
			
			start=$((start+length))
		done
	done
done
