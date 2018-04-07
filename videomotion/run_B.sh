#!/bin/bash

# ---------------------------------- #
expid=stability_reality1

videopath=../data/
videoname=skater.avi
rep=100

outpath=exp2/

features=5
filtersize=5

theta=0.0001
alpha=7.8125
beta=0.000000025
gamma=0.000375
k=0.0000000000000000025

step=0.01

eta=0.001
rho=0
blur=1

lambdaC=1000
lambdaE=2000

eps1=150
eps2=150
eps3=150

mkdir ${outpath}/test

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0

# ---------------------------------- #
expid=stability_reality2
theta=0.0001
alpha=10253.8
beta=0.0000728553
gamma=0.864277
k=0.0000000000000022456

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0

# ---------------------------------- #
expid=stability_notreality1
theta=0.0001
alpha=78.1303
beta=0.000000250141
gamma=0.00375141
k=0.000000000000000337498

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0

# ---------------------------------- #
expid=stability_notreality2
theta=0.0001
alpha=3600000000
beta=1
gamma=60000
k=0.000000018

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0

# ---------------------------------- #
expid=notstability_reality1
theta=0.0001
alpha=5
beta=2
gamma=1
k=0.09999

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0

# ---------------------------------- #
expid=notstability_reality2
theta=0.0001
alpha=0.049995
beta=2
gamma=1
k=20

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0

# ---------------------------------- #
expid=notstability_reality3
theta=0.0001
alpha=2.5
beta=12.6491
gamma=0
k=16

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0

# ---------------------------------- #
expid=notstability_notreality1
theta=0.0001
alpha=1
beta=0.000000000625
gamma=0.00000125
k=0.001

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0

# ---------------------------------- #
expid=notstability_notreality2
theta=0.0001
alpha=1
beta=1
gamma=1
k=10

expname=${expid}$"_"${videoname}"_"rep${rep}"_"m${features}"_"f${filtersize}"_"theta${theta}"_"alpha${alpha}"_"beta${beta}"_"gamma${gamma}"_"k${k}"_"step${step}"_"eta${eta}"_"rho${rho}"_"blur${blur}"_"lambdaC${lambdaC}"_"lambdaE${lambdaE}"_"eps1${eps1}"_"eps2${eps2}"_"eps3${eps3}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/${expname} --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep ${rep} --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho ${rho} --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size ${step} --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
cp -r ${outpath}/${expname} ${outpath}/test/${expname}
python vprocessor.py --run ${videopath}${videoname} --out ${outpath}/test/${expname} --resume 2 --port 0 --gray 1 --save_scores_only 0 --res 240x110 --day_only 0 --check_params 0 --rep 1 --all_black 0 --grad 0 --gew 1.0 --m ${features} --f ${filtersize} --init_fixed 0 --init_q 1.0 --alpha ${alpha} --beta ${beta} --gamma ${gamma} --k ${k} --theta ${theta} --eta ${eta} --rho 1 --eps1 ${eps1} --eps2 ${eps2} --eps3 ${eps3} --step_size 0 --step_adapt 0 --lambdaM 0.0 --lambdaE ${lambdaE} --lambdaC ${lambdaC} --blur ${blur} --grad_order2 0
