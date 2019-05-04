# OPENCV_SGBM
Demo stereo matching converting L and R images into disparity, point clouds and PLY model

Usage: stereo_match <left_image> <right_image>
[--blocksize=<block_size>] [--min-disparity=<min_disparity>]
[--max-disparity=<max_disparity>] [--scale=scale_factor>]
[--no-display]

Example: stereo_match.exe left_image.jpg right_image.jpg --blocksize=5 --min-disparity=16 --max-disparity=192 --scale=0.5 

Stereo data here: http://vision.middlebury.edu/stereo/data/

You can open result PLY file by MeshLab: http://www.meshlab.net/
