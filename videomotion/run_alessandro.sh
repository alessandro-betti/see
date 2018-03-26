# With this parameters the day-equation (without the input) is stable, yet there is a strange phenomenon at the
# transition with the night: The third derivative get enormously big (numerical error?).
#python vprocessor.py --run data/skater.avi --out exp/skater1\
#       --gray 1 --save_scores_only 0 --res 240x180 --day_only 0 --check_params 1 --rep 100 --all_black 0 --grad 0 --port 8888 --gew 1.0\
#       --m 3 --f 3\
#       --init_fixed 1 --init_q 1.0\
#       --alpha 1 --beta  1 --gamma 30001 --k 1 --theta 0.0001\
#       --eta 0 --rho 0\
#       --alpha_night 0.001 --beta_night 1.0 --gamma_night 1.0 --thetanight 10000.0\
#       --eps1 1 --eps2 1 --eps3 1000 --zeta 0.001\
#       --step_size 0.01 --step_size_night 0.000001 --step_adapt 0\
#       --lambda1 0.0 --lambda0 0.0 --softmax 1\
#       --lambdaM 0.0 --lambdaE 200.0 --lambdaC 100.0


# Yet another experiment with crazy coefficients lambdaCE and lambdaC in order to see the effects of the
# information-based constraint.
#python vprocessor.py --run data/skater.avi --out exp/skater1\
#       --gray 1 --save_scores_only 0 --res 240x180 --day_only 1 --check_params 1 --rep 100 --all_black 0 --grad 0 --port 8888 --gew 1.0\
#       --m 3 --f 3\
#       --init_fixed 0 --init_q 1.0\
#       --alpha 0.1 --beta  0.001 --gamma 633 --k 0.001 --theta 0.0001\
#       --eta 0.0001 --rho 0\
#       --alpha_night 0.001 --beta_night 1.0 --gamma_night 1.0 --thetanight 10000.0\
#       --eps1 1 --eps2 1 --eps3 1000 --zeta 0.001\
#       --step_size 0.01 --step_size_night 0.000001 --step_adapt 0\
#       --lambda1 0.0 --lambda0 0.0 --softmax 1\
#       --lambdaM 0.0 --lambdaE 20000.0 --lambdaC 10000.0


# Trying to reproduce gradient decent with appropriate settings
# The idea here is that when \alpha=\beta=k=0 and \gamma~1/\theta^2 with \theta>>1 we reproduce the classical
# gradient descent
python vprocessor.py --run data/skater.avi --out exp/skater1\
       --gray 1 --save_scores_only 0 --res 240x180 --day_only 1 --check_params 0 --rep 100 --all_black 0 --grad 0 --port 8888 --gew 1.0\
       --m 3 --f 3\
       --init_fixed 0 --init_q 1.0\
       --alpha 0.000001 --beta  0 --gamma 0.01 --k 0 --theta 10\
       --eta 0.0001 --rho 1\
       --alpha_night 0.001 --beta_night 1.0 --gamma_night 1.0 --thetanight 10000.0\
       --eps1 1 --eps2 1 --eps3 1000 --zeta 0.001\
       --step_size 0.001 --step_size_night 0.000001 --step_adapt 0\
       --lambda1 0.0 --lambda0 0.0 --softmax 1\
       --lambdaM 0.0 --lambdaE 2.0 --lambdaC 1.0

