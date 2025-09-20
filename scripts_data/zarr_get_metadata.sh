# bash scripts_data/zarr_get_metadata.sh

input_dir="/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data/zarr_data_egodex_subset0.15|\
/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data_gpt/zarr_data_h2o|\
/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data_gpt/zarr_data_hoi4d|\
/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data_gpt/zarr_data_human|\
/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data_gpt/zarr_data_taco_subset0.6|\
/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data_gpt/zarr_data_robot
"
output_dir="/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic"


python -m scripts_data.entry.zarr_get_metadata \
    --input_dir ${input_dir} \
    --output_dir ${output_dir} 