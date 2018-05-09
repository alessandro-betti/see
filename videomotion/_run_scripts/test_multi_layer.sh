#!/bin/bash

# ---------------------------------- #
# SHARED OPTIONS
#
# Run 'vprocessor.py' without any arguments to see the full option list (and description)
#
# Warning: there are some naming conventions that differ from the ones used in the paper and the ones used in the code
# and in the bash scripts (sh)
#
#    paper       code / sh
#    --------------------------
#    n           m / features
#    f           f / filtersize
#    \mu         alpha / alpha
#    \nu         beta / beta
#    h           step_size / step
#    \phi(0)     rho / rho
# ---------------------------------- #
videopath=data/
videoname=skater.avi   # skater.avi, 228 frames

outpath=test_multi_layer/

step=0.04  # 25 fps

features=5
filtersize=5

eta=0.0005
rho=0
blur=1

eps1=300
eps2=300
eps3=300

# multi-layer options
layers=3  # 3 layers
rep=600  # 200 repetitions per layer
c_frames_min=45600  # this is the number of frames for 200 repetitions of the input video
c_frames=45600  # this is the number of frames for 200 repetitions of the input video

lambdaC=1000
lambdaE_0=2000
lambdaE_1=5000  # it seems that higher layers require larger lambdaEs...maybe it is due to the fact that each feature processes multiple input channels?
lambdaE_2=5000  # it seems that higher layers require larger lambdaEs...maybe it is due to the fact that each feature processes multiple input channels?

c_eps1=1000000
c_eps2=1000000
c_eps3=1000000

mkdir -p ${outpath}
mkdir -p ${outpath}/test

# ---------------------------------- #
# EXPERIMENTS
# ---------------------------------- #
expid=stability_reality
theta=0.0001
alpha=7.8125
beta=0.00000003125
gamma=0.000375
k=0.000000000000000000625

expname=${expid}"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE_0}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 1 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE_0 ${lambdaE_0} --lambdaE_1 ${lambdaE_1} --lambdaE_2 ${lambdaE_2} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0 --layers ${layers} --c_frames_min ${c_frames_min} --c_frames ${c_frames} --c_eps1 ${c_eps1} --c_eps2 ${c_eps2} --c_eps3 ${c_eps3}
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE_0 ${lambdaE_0} --lambdaE_1 ${lambdaE_1} --lambdaE_2 ${lambdaE_2} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0 --layers ${layers} --c_frames_min ${c_frames_min} --c_frames ${c_frames} --c_eps1 ${c_eps1} --c_eps2 ${c_eps2} --c_eps3 ${c_eps3}
