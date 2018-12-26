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
videoname=skater.avi

outpath=fig1_column1/

rep=200
step=0.04

features=5
filtersize=5

eta=0.0005
rho=0
blur=1

lambdaC=1000
lambdaE=2000

eps1=300
eps2=300
eps3=300

mkdir ${outpath}
mkdir ${outpath}/test

# ---------------------------------- #
# EXPERIMENTS
# ---------------------------------- #
expid=stability_reality
theta=0.0001
alpha=7.8125
beta=0.00000003125
gamma=0.000375
k=0.000000000000000000625

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 1 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0

# ---------------------------------- #
expid=stability_notreality
theta=0.0001
alpha=78.1303
beta=0.000000250141
gamma=0.00375141
k=0.000000000000000337498

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 1 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0

# ---------------------------------- #
expid=notstability_reality
theta=0.0001
alpha=9.969
beta=0.000638395
gamma=0.0790333
k=0.00000000996891

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 1 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0

# ---------------------------------- #
expid=notstability_notreality
theta=0.0001
alpha=1
beta=0.000000000625
gamma=0.00000125
k=0.001

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 1 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python ../vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
