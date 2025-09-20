# bash scripts_data/data_visualization.sh

data_path="data/zarr_data/zarr_data_human/human_me+drop_bread_to_the_green_bucket+.zarr"
episode_idx=0
downsample_factor=1

python -m scripts_data.entry.data_visualization \
  --data_path "${data_path}" \
  --episode_idx ${episode_idx} \
  --downsample_factor ${downsample_factor} \
  --precheck_pointclouds                     # pre-check the overlayed pointclouds