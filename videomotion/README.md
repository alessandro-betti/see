# TNNLS
This folder contains the source code used to run the experiments of the paper.
 
Scripts and data to reproduce these experiments are in the [tnnls](tnnls) folder.
 
# vprocessor
The main file is vprocessor.py that depends on [TensorFlow](https://www.tensorflow.org) 
and [OpenCV](https://opencv.org).

Launching

	python vprocessor.py
	
one gets the following output Usage:

	vprocessor.py --run <file/folder/0> --out <folder> [options]

where 'file' is video file, and '0' indicates the local web-cam.

## Options
Some options can be:
* --res <number>x<number>: custom video resolution;
* --fps <number>: custom video frame rate;
* --frames <number>: maximum number of frames to process;
* --m <number>: number of features;
* --f <number>: filter size (e.g., 3 for 3x3 filters);
* --gray <1/0>: Put the imput to zero;
* --init_q <number>: Initialize the filters with standard deviation <number>;
* --init_fixed <number>: Initialize the filters all to the same <number> value;
	
The full list of options can be visualized by launching 

	vprocessor.py 

Other options have been added to modify the values of the hyperparamenters:

	--k <number> --alpha <number> --theta <number> --beta <number> --gamma <number> --eta <number> --eps1 <number> --eps2 <number>...

### Examples of command line
	python vprocessor.py --run data/skater.avi --out exp/skater --m 3 --f 3 --gray 1 --res 100x80

	python vprocessor.py --run data/skater.avi --out exp/skater --m 10 --f 3 --gray 1 --res 100x80 --theta 0.00001 --k 100 --beta 3000 --gamma=20000000000 --alpha 10000000 --rep 100000

## Visualize output
In order to visualize the feature developed one can access the web service at http://127.0.0.1:port 
where the port number is showned when running the experiment.

When the experiment is terminated the web service can be opened using

	python visuserver.py exp/skater

Also a tensor-board visualization can be launched with

	tensorboard --logdir=exp/skater/tensor_board

and can be seen at http://127.0.0.1:6006

