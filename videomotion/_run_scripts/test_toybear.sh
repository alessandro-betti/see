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
videoname=toybearblack.avi

outpath=test_toybear/

step=0.04  # 25 fps

features=3
filtersize=5

eta=0.005
rho=0
blur=1

eps1=3000
eps2=3000
eps3=3000

# multi-layer options
layers=1
rep=100

lambdaC=100
lambdaE=500
lambdaM=0.0001

mkdir -p ${outpath}
#mkdir -p ${outpath}/test

# ---------------------------------- #
# EXPERIMENTS
# ---------------------------------- #
expid=stability_reality
theta=0.0001
alpha=7.8125
beta=0.00000003125
gamma=0.000375
k=0.000000000000000000625

expname=toybear
python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --resume 0 --port 8888 --res 120x120 --gray 1 --save_scores_only 0 --day_only 1 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --lambdaM ${lambdaM} --blur ${blur} --grad_order2 0 --layers ${layers}
#cp -r ${outpath}/${expname} ${outpath}/test/${expname}
#python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 8888 --res 125x100 --gray 1 --save_scores_only 0 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --lambdaM ${lambdaM} --blur ${blur} --grad_order2 0 --layers ${layers}
