# bash scripts_data/zarr_get_subset.sh

# dataset_path="/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data/zarr_data_egodex"
# ratio=0.15

dataset_path="/cephfs/shared/yuanchengbo/hub/huggingface/dexmimic/zarr_data_gpt/zarr_data_taco"
ratio=0.6

python -m scripts_data.entry.zarr_get_subset \
    --dataset_path ${dataset_path} \
    --ratio ${ratio} \