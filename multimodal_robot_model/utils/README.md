# Utilities

## Data processing
### Visualize npz file
```console
$ python ./visualize_data.py <npz file>
```

### Convert rosbag file to npz file
```console
$ python ./convert_rosbag_to_npz.py <rosbag directory>
```

### Trim npz file
```console
$ python ./trim_npz.py <npz directory>
```

### Trim npz file
```console
$ python tile_teleop_videos.py <output filename> <npz directory> --column_num 2 --envs env0 env1 env4 env5
```
The options `--column_num` and `--envs` can be omitted.
