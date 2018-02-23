# vprocessor

The main file is vprocessor.py

Launching
python vprocessor.py
one gets the following output Usage:

vprocessor.py --run <file/folder/0> --out <folder> [options]
vprocessor.py --pre <file/0> --out <folder> [options]

where 'file' is video file, and '0' indicates the local web-cam

Options can be:
	--res <number>x<number>: custom video resolution
	--fps <number>: custom video frame rate
	--frames <number>: maximum number of frames to process
	--m <number>: number of features
	--f <number>: filter size (e.g., 3 for 3x3 filters)

