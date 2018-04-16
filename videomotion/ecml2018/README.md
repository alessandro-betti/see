# ECML 2018

This folder contains the scripts used to reproduce the experiments described in the paper.

Scripts will access the data located in the "data" subfolder; each script will create multiple 
subfolders inside the output folder (see the description of [vprocessor](https://github.com/alessandro-betti/see/blob/master/videomotion/README.md#vprocessor)) with the outcomes of the experiments.

Similar graphs to the ones in the paper can be visualized by 

    tensorboard --logdir X/tensor_board

where X is one of the previousely mentioned folders.

The results in Table~1 are taken from the output folder test/X, and they correspond to the 
the values computed on the last frame.

