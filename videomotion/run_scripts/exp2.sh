# Exp 2
python ../vprocessor.py --run ../data/skater.avi --out ../exp/skater1\
       --gray 1 --save_scores_only 0 --res 240x180 --day_only 0 --check_params 0 --rep 100000000\
       --all_black 0 --grad 0 --port 8888 --gew 1.0\
       --m 6 --f 5\
       --init_fixed 0 --init_q 1.0\
       --alpha 1.0 --beta 0.00000000625 --gamma 0.0000125 --k 0.001 --theta 0.0001\
       --eta 0.0001 --rho 0\
       --alpha_night 0.001 --beta_night 1.0 --gamma_night 1.0 --thetanight 10000.0\
       --eps1 1000 --eps2 1000 --eps3 1000 --zeta 0.001\
       --step_size 0.1 --step_size_night 0.0000001 --step_adapt 0\
       --lambda1 0.0 --lambda0 0.0 --softmax 1\
       --lambdaM 0.0 --lambdaE 20000.0 --lambdaC 10000.0
