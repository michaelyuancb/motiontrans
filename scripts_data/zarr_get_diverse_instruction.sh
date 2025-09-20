# bash scripts_data/zarr_get_diverse_instruction.sh

input_dir="/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data/zarr_data_human"
output_dir="/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data_gpt"
embodiment="human"

# input_dir="/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data/zarr_data_robot"
# output_dir="/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data_gpt"
# embodiment="robot"


python -m scripts_data.entry.zarr_get_diverse_instruction \
    --input_dir ${input_dir} \
    --output_dir ${output_dir} \
    --embodiment ${embodiment}