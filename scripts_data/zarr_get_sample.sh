# bash scripts_data/zarr_get_sample.sh

replay_buffer_fp="/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data/zarr_data_human/human_me+wipe_blue_towel_to_the_bulky_bottle+.zarr"
output_dir="."
num_use_demo=1


python -m scripts_data.entry.zarr_get_sample \
    --replay_buffer_fp ${replay_buffer_fp} \
    --output_dir ${output_dir} \
    --num_use_demo ${num_use_demo} \