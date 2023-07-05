#!/bin/bash

### Mixing N models/conditionings to generate a mix of styles. 1. Model A trained w.o. style conditioning 2. Model B with style S1
### 3. Model B with style S2. The sampling will be done with e=e0 + g1*(e1-e0) + g2*(e2-e0) + ...
### gn are guidance factors. E.g. g1=1, g2=0 => style S1 g1=0,g2=1 => S2, g1=1,g2=1 => mix of styles

# # Dance
checkpoint1=pretrained_models/dance_LDA-U.ckpt
checkpoint2=pretrained_models/dance_LDA.ckpt
checkpoint3=${checkpoint2}
dest_dir=results/dance_mix_experts

if [ ! -d "${dest_dir}" ]; then
    mkdir -p "${dest_dir}"
fi

data_dir=data/motorica_dance
basenames=$(cat "${data_dir}/gen_files.txt")

# Different guidance factors for mixing models
guidance_factors_lst=("1.0,1.0" "0.5,0.5" "0.25,1.0" "1.0,0.25")
style=None,gJZ,gLO

start=0
seed=150
fps=30
trim_s=0
length_s=10
trim=$((trim_s*fps))
length=$((length_s*fps))
fixed_seed=false
gpu="cuda:0"
render_video=true

for wavfile in $basenames;
do
	start=0
	for postfix in 0
	do		
		for guidance_factors in ${guidance_factors_lst[@]}; do
	
			input_file=${wavfile}.audio29_${fps}fps.pkl			
			input_file2=${input_file}
			input_file3=${input_file}					
			
			output_file=${wavfile::-3}_${postfix}_${style}_${guidance_factors}
			
			echo Generating motion from ${input_file} to ${dest_dir}/${output_file}		
			echo "start=${start}, len=${length}, postfix=${postfix}, seed=${seed}"

			python synthesize.py --checkpoints="${checkpoint1},${checkpoint2},${checkpoint3}" --data_dirs="${data_dir},${data_dir},${data_dir}" --input_files="${input_file},${input_file2},${input_file3}" --styles="${style}" --start=${start} --end=${length} --trim=${trim} --seed=${seed} --postfix=${postfix} --dest_dir=${dest_dir} --gf=${guidance_factors} --gpu=${gpu} --outfile=${output_file} --video=${render_video}
			if [ "$fixed_seed" != "true" ]; then
				seed=$((seed+1))
			fi
			echo seed=$seed
			python utils/cut_wav.py ${data_dir}/${wavfile::-3}.wav $(((start+trim)/fps)) $(((start+length-trim)/fps)) ${postfix} ${dest_dir}
			ffmpeg -y -i ${dest_dir}/${output_file}.mp4 -i ${dest_dir}/${wavfile::-3}_${postfix}.wav ${dest_dir}/${output_file}_audio.mp4
			rm ${dest_dir}/${output_file}.mp4
			
			start=$((start+length))
			postfix=$((postfix+1))
		done
	done
done
