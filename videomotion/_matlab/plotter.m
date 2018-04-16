clear variables;
close all;
%j=6; [xm,ym] = prepare_data(x,y,j);
%plotfig(xm(1:1000),ym(1,1:1000),explabels,measure_labels{j},figdest,measure_file_labels{j});
%    dfdsfsdfs
%for j = 1:length(tensorboard_measures)
%    [xm,ym] = prepare_data(x,y,j);
%    plotfig(xm,ym,explabels,measure_labels{j},figdest);
%end
%dfdsfsd

% ----------------
% CONFIGURATION
% ----------------
figdest_small_k = '/Users/mela/Desktop/ECML/Paper/fig/small_k/';
color_pattern = 1;
expdirsA_small_k = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_notreality1_skater.avi_rep200_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_notreality2_skater.avi_rep200_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_reality1_skater.avi_rep200_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_reality2_skater.avi_rep200_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_reality3_skater.avi_rep200_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_reality4_skater.avi_rep200_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/stability_notreality1_skater.avi_rep200_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/stability_notreality2_skater.avi_rep200_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/stability_reality1_skater.avi_rep200_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/stability_reality2_skater.avi_rep200_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/stability_reality3_skater.avi_rep200_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    };
expdirsB_small_k = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_notreality1_skater.avi_rep500_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_notreality2_skater.avi_rep500_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_reality1_skater.avi_rep500_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_reality2_skater.avi_rep500_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_reality3_skater.avi_rep500_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_reality4_skater.avi_rep500_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/stability_notreality1_skater.avi_rep500_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/stability_notreality2_skater.avi_rep500_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/stability_reality1_skater.avi_rep500_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/stability_reality2_skater.avi_rep500_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/stability_reality3_skater.avi_rep500_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...   
    };
expdirsC_small_k = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_notreality1_skater.avi_rep50_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_notreality2_skater.avi_rep50_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_reality1_skater.avi_rep50_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_reality2_skater.avi_rep50_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_reality3_skater.avi_rep50_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_reality4_skater.avi_rep50_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/stability_notreality1_skater.avi_rep50_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/stability_notreality2_skater.avi_rep50_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/stability_reality1_skater.avi_rep50_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/stability_reality2_skater.avi_rep50_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/stability_reality3_skater.avi_rep50_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300'    
    };
testexpdirsA_small_k = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_notreality1_skater.avi_rep200_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_notreality2_skater.avi_rep200_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_reality1_skater.avi_rep200_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_reality2_skater.avi_rep200_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_reality3_skater.avi_rep200_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_reality4_skater.avi_rep200_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/stability_notreality1_skater.avi_rep200_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/stability_notreality2_skater.avi_rep200_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/stability_reality1_skater.avi_rep200_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/stability_reality2_skater.avi_rep200_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/stability_reality3_skater.avi_rep200_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    };
testexpdirsB_small_k = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_notreality1_skater.avi_rep500_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_notreality2_skater.avi_rep500_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_reality1_skater.avi_rep500_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_reality2_skater.avi_rep500_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_reality3_skater.avi_rep500_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_reality4_skater.avi_rep500_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/stability_notreality1_skater.avi_rep500_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/stability_notreality2_skater.avi_rep500_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/stability_reality1_skater.avi_rep500_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/stability_reality2_skater.avi_rep500_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/stability_reality3_skater.avi_rep500_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...   
    };
testexpdirsC_small_k = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_notreality1_skater.avi_rep50_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_notreality2_skater.avi_rep50_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_reality1_skater.avi_rep50_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_reality2_skater.avi_rep50_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_reality3_skater.avi_rep50_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_reality4_skater.avi_rep50_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/stability_notreality1_skater.avi_rep50_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/stability_notreality2_skater.avi_rep50_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/stability_reality1_skater.avi_rep50_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/stability_reality2_skater.avi_rep50_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/stability_reality3_skater.avi_rep50_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300'    
    };
figdest_large_k = '/Users/mela/Desktop/ECML/Paper/fig/large_k/';
expdirsA_large_k = {
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_notreality1_skater.avi_rep200_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_notreality2_skater.avi_rep200_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_reality1_skater.avi_rep200_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_reality2_skater.avi_rep200_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_reality3_skater.avi_rep200_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/notstability_reality4_skater.avi_rep200_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/stability_notreality1_skater.avi_rep200_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/stability_notreality2_skater.avi_rep200_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/stability_reality1_skater.avi_rep200_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/stability_reality2_skater.avi_rep200_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/stability_reality3_skater.avi_rep200_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    };
expdirsB_large_k = {
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_notreality1_skater.avi_rep500_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_notreality2_skater.avi_rep500_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_reality1_skater.avi_rep500_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_reality2_skater.avi_rep500_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_reality3_skater.avi_rep500_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/notstability_reality4_skater.avi_rep500_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/stability_notreality1_skater.avi_rep500_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/stability_notreality2_skater.avi_rep500_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/stability_reality1_skater.avi_rep500_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/stability_reality2_skater.avi_rep500_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/stability_reality3_skater.avi_rep500_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...   
    };
expdirsC_large_k = {
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_notreality1_skater.avi_rep50_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_notreality2_skater.avi_rep50_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_reality1_skater.avi_rep50_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_reality2_skater.avi_rep50_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_reality3_skater.avi_rep50_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/notstability_reality4_skater.avi_rep50_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/stability_notreality1_skater.avi_rep50_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/stability_notreality2_skater.avi_rep50_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/stability_reality1_skater.avi_rep50_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/stability_reality2_skater.avi_rep50_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/stability_reality3_skater.avi_rep50_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300'    
    };
testexpdirsA_large_k = {
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_notreality1_skater.avi_rep200_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_notreality2_skater.avi_rep200_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_reality1_skater.avi_rep200_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_reality2_skater.avi_rep200_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_reality3_skater.avi_rep200_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/notstability_reality4_skater.avi_rep200_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/stability_notreality1_skater.avi_rep200_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/stability_notreality2_skater.avi_rep200_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/stability_reality1_skater.avi_rep200_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/stability_reality2_skater.avi_rep200_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expA/test/stability_reality3_skater.avi_rep200_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    };
testexpdirsB_large_k = {
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_notreality1_skater.avi_rep500_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_notreality2_skater.avi_rep500_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_reality1_skater.avi_rep500_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_reality2_skater.avi_rep500_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_reality3_skater.avi_rep500_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/notstability_reality4_skater.avi_rep500_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/stability_notreality1_skater.avi_rep500_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/stability_notreality2_skater.avi_rep500_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/stability_reality1_skater.avi_rep500_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/stability_reality2_skater.avi_rep500_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expB/test/stability_reality3_skater.avi_rep500_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.1_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...   
    };
testexpdirsC_large_k = {
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_notreality1_skater.avi_rep50_m5_f5_theta0.0001_alpha1_beta0.000000000625_gamma0.00000125_k0.001_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_notreality2_skater.avi_rep50_m5_f5_theta0.0001_alpha1_beta1_gamma1_k10_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_reality1_skater.avi_rep50_m5_f5_theta0.0001_alpha5_beta2_gamma1_k0.09999_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_reality2_skater.avi_rep50_m5_f5_theta0.0001_alpha0.049995_beta2_gamma1_k20_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_reality3_skater.avi_rep50_m5_f5_theta0.0001_alpha2.5_beta12.6491_gamma0_k16_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/notstability_reality4_skater.avi_rep50_m5_f5_theta0.0001_alpha9.969_beta0.000638395_gamma0.0790333_k0.00000000996891_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/stability_notreality1_skater.avi_rep50_m5_f5_theta0.0001_alpha78.1303_beta0.000000250141_gamma0.00375141_k0.000000000000000337498_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/stability_notreality2_skater.avi_rep50_m5_f5_theta0.0001_alpha3600000000_beta1_gamma60000_k0.000000018_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    
    %%%%%%%'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/stability_reality1_skater.avi_rep50_m5_f5_theta0.0001_alpha7.8125_beta0.000000025_gamma0.000375_k0.0000000000000000025_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/stability_reality2_skater.avi_rep50_m5_f5_theta0.0001_alpha10253.8_beta0.0000728553_gamma0.864277_k0.0000000000000022456_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    %'/Users/mela/Desktop/ECML/Experiments/maxi/expC/test/stability_reality3_skater.avi_rep50_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.01_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300'    
    };
figdest5x5 = '/Users/mela/Desktop/ECML/Paper/fig/5x5/';
color_patternf = 2;
expdirs5x5 = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp5x5_5/stability_reality3_car.avi_rep180_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp5x5_5/stability_reality3_matrix.avi_rep50_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp5x5_5/stability_reality3_skater.avi_rep200_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300'
    };
testexpdirs5x5 = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp5x5_5/test/stability_reality3_car.avi_rep180_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp5x5_5/test/stability_reality3_matrix.avi_rep50_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp5x5_5/test/stability_reality3_skater.avi_rep200_m5_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300'
    };
figdest11x11 = '/Users/mela/Desktop/ECML/Paper/fig/11x11/';
expdirs11x11 = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp11x11_11/stability_reality3_car.avi_rep180_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp11x11_11/stability_reality3_matrix.avi_rep50_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp11x11_11/stability_reality3_skater.avi_rep200_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300';
    };
testexpdirs11x11 = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp11x11_11/test/stability_reality3_car.avi_rep180_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp11x11_11/test/stability_reality3_matrix.avi_rep50_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/exp11x11_11/test/stability_reality3_skater.avi_rep200_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300';
    };
figdestblur = '/Users/mela/Desktop/ECML/Paper/fig/blur/';
color_patternb = 3;
expdirsblur = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/expNightBlur/stability_reality3_skater.avi_rep200_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.00001_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expNightBlur/stability_reality3_skater.avi_rep200_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expNightBlur/stability_reality3_skater.avi_rep200_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho1_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300'
    };
testexpdirsblur = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/expNightBlur/test/stability_reality3_skater.avi_rep200_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.00001_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expNightBlur/test/stability_reality3_skater.avi_rep200_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expNightBlur/test/stability_reality3_skater.avi_rep200_m11_f11_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho1_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300'
    };
figdestgrad = '/Users/mela/Desktop/ECML/Paper/fig/grad/';
color_patterng = 4;
expdirsgrad = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/expGrad/stability_reality3_grad_order2_skaterc.avi_rep65_m10_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expGrad/stability_reality3_skaterc.avi_rep65_m10_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300'    
    };
testexpdirsgrad = {
    '/Users/mela/Desktop/ECML/Experiments/maxi/expGrad/test/stability_reality3_grad_order2_skaterc.avi_rep65_m10_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300', ...
    '/Users/mela/Desktop/ECML/Experiments/maxi/expGrad/test/stability_reality3_skaterc.avi_rep65_m10_f5_theta0.0001_alpha7.8125_beta0.00000003125_gamma0.000375_k0.000000000000000000625_step0.04_eta0.0005_rho0_blur1_lambdaC1000_lambdaE2000_eps1300_eps2300_eps3300'    
    };

explabels = {
    'no-stability, no-reality', ...
    'no-stability, reality', ...
    'stability, no-reality', ...
    'stability, reality'
    };

explabelsb = {
    'slow', ...
    'fast', ...
    'faster (none)'
    };

explabelsv = {
    'car', ...
    'matrix', ...
    'skater'
    };

explabelsg = {
    'gradient', ...
    'cognitive laws'
    };

fixg = false;
no_y_label = true;

%figdest = [figdest_small_k]; expdirs = expdirsA_small_k; testexpdirs = testexpdirsA_small_k;
%figdest = [figdest_small_k 'faster/']; expdirs = expdirsB_small_k; testexpdirs = testexpdirsB_small_k;
%figdest = [figdest_small_k 'slower/']; expdirs = expdirsC_small_k; testexpdirs = testexpdirsC_small_k;

%figdest = [figdest_large_k]; expdirs = expdirsA_large_k; testexpdirs = testexpdirsA_large_k;
%%%%%%figdest = [figdest_large_k 'faster/']; expdirs = expdirsB_large_k; testexpdirs = testexpdirsB_large_k;
%%%%%%figdest = [figdest_large_k 'slower/']; expdirs = expdirsC_large_k; testexpdirs = testexpdirsC_large_k;

%figdest = [figdest5x5]; expdirs = expdirs5x5; testexpdirs = testexpdirs5x5; explabels = explabelsv;  color_pattern = color_patternf;
%%%%%%figdest = [figdest11x11]; expdirs = expdirs11x11; testexpdirs = testexpdirs11x11; explabels = explabelsv;
%figdest = [figdest11x11]; expdirs = strrep(strrep(expdirs11x11,'exp11x11_11','exp11x11_11bis'),'lambdaE2000','lambdaE4000'); testexpdirs = strrep(strrep(testexpdirs11x11,'exp11x11_11','exp11x11_11bis'),'lambdaE2000','lambdaE4000'); explabels = explabelsv;  color_pattern = color_patternf;

%%figdest = [figdestblur]; expdirs = expdirsblur; testexpdirs = testexpdirsblur; explabels = explabelsb;
%%%%%%%figdest = [figdestblur]; expdirs = strrep(strrep(expdirsblur,'expNightBlur','expNightBlurbis'),'lambdaE2000','lambdaE4000'); testexpdirs = strrep(strrep(testexpdirsblur,'expNightBlur','expNightBlurbis'),'lambdaE2000','lambdaE4000'); explabels = explabelsb;  color_pattern = color_patternb;
figdest = [figdestblur]; expdirs = strrep(strrep(strrep(expdirsblur,'expNightBlur','expNightBlurbis2'),'lambdaE2000','lambdaE4000'),'eta0.00001','eta0.0001'); testexpdirs = strrep(strrep(strrep(testexpdirsblur,'expNightBlur','expNightBlurbis2'),'lambdaE2000','lambdaE4000'),'eta0.00001','eta0.0001'); explabels = explabelsb; color_pattern = color_patternb;

%%%%%%figdest = [figdestgrad]; expdirs = expdirsgrad; testexpdirs = testexpdirsgrad; explabels = explabelsg;
%figdest = [figdestgrad]; expdirs = strrep(strrep(strrep(expdirsgrad,'expGrad','expGradbis'),'lambdaE2000','lambdaE4000'),'_rep200','_rep65'); testexpdirs = strrep(strrep(strrep(testexpdirsgrad,'expGrad','expGradbis'),'lambdaE2000','lambdaE4000'),'_rep200','_rep65'); explabels = explabelsg; fixg = true; color_pattern = color_patterng;

density_window = 1000; % if <= 1 then 'moving average', if < 0 then 'gaussian smoothing'
tensorboard_conv_script = './exportTensorFlowLog.py';
tensorboard_measures = {
    'main/AC_RealMutualInformationFull', ...    
    'main/BC_RealMutualInformation', ...
    'main/BG_NormQ', ...
    'main/CC_CompleteRealMutualInformation', ...
    'main/CG_CompleteNormQ', ...    
    'main/AA_Night', ...
    'main/BA_CognitiveAction', ...    
    'main/CA_CompleteCognitiveAction', ...    
    'main/AB_Rho'
    };
measure_labels = {
    'MI (global)', ...    
    'MI', ...
    'Squared Norm of q', ...
    'CAL-MI', ...
    'CAL-Squared Norm of q', ...    
    'Resets', ...
    'Cognitive Action', ...
    'CAL-Cognitive Action', ...
    'Blurring Factor'
    };
measure_file_labels = {
    'mifull', ...
    'mi', ...
    'normq', ...
    'ca-mi', ...
    'ca-normq', ...    
    'blinks', ...
    'action', ...
    'ca-action', ...
    'rho'
    };
if no_y_label == true
    for i = 1:length(measure_file_labels)
        measure_file_labels{i} = [measure_file_labels{i} '_noylabel'];
    end
end
% ----------------

% reading TensorBoard data
ne = length(expdirs);
CSV_header = cell(ne,1);
CSV = cell(ne,1);
for i = 1:ne
    tensorboarddir = [expdirs{i} '/tensor_board/'];
    tensorboardcsv = [tensorboarddir '/scalars.csv'];
    if ~exist(tensorboardcsv, 'file')
        system(['python ' tensorboard_conv_script ' ' tensorboarddir ' ' tensorboarddir]);
    end
    fid = fopen(tensorboardcsv);
    header_line = fgetl(fid);
    fclose(fid);
    CSV_header{i} = strsplit(header_line,',');
    CSV{i} = csvread(tensorboardcsv,1);
    assert(CSV{i}(end,2) == size(CSV{i},1), 'Invalid CSV: the last step is different from the number of samples!');
end

% finding the TensorBoard data we need to plot
measure_cols = -1 * ones(ne,length(tensorboard_measures),1);
for k = 1:ne
    for i = 1:length(CSV_header{k})
        for j = 1:length(tensorboard_measures)
            if strcmp(CSV_header{k}{i}, tensorboard_measures{j})
                measure_cols(k,j) = i;
            end
        end  
    end
end
assert(min(measure_cols(:)) > 0, 'Cannot find some data...');

% getting TensorBoard data
x = cell(ne,1);
y = cell(ne,length(tensorboard_measures));
for i = 1:ne
    x{i} = CSV{i}(:,2);
    for j = 1:length(tensorboard_measures)
        y{i,j} = CSV{i}(:,measure_cols(i,j));
        if length(unique(y{i,j})) == 2
            y{i,j} = to_density(y{i,j},density_window);
        end
    end
end

% reading the blink data
%blinks = cell(ne,1);
%for i = 1:ne
%    infofile = [expdirs{i} '/model/model.saved.info.txt'];
%    jsondata = jsondecode(fileread(infofile));
%    blinks{i} = zeros(t{i},1);
%    blinks{i}(jsondata.blink_steps) = 1;
%end

% reading TensorBoard data (test run)
CSVtest_header = cell(ne,1);
CSVtest = cell(ne,1);
for i = 1:ne
    tensorboarddir = [testexpdirs{i} '/tensor_board/'];
    tensorboardcsv = [tensorboarddir '/scalars.csv'];
    if ~exist(tensorboardcsv, 'file')
        system(['python ' tensorboard_conv_script ' ' tensorboarddir ' ' tensorboarddir]);
    end
    fid = fopen(tensorboardcsv);
    header_line = fgetl(fid);
    fclose(fid);
    CSVtest_header{i} = strsplit(header_line,',');    
    CSVtest{i} = csvread(tensorboardcsv,1);
    assert(CSVtest{i}(end,2) == size(CSVtest{i},1), 'Invalid CSV (test): the last step is different from the number of samples!');
end

% finding the TensorBoard data we need to plot (test run)
measure_cols = -1 * ones(ne,length(tensorboard_measures),1);
for k = 1:ne
    for i = 1:length(CSVtest_header{k})
        for j = 1:length(tensorboard_measures)
            if strcmp(CSVtest_header{k}{i}, tensorboard_measures{j})
                measure_cols(k,j) = i;
            end
        end  
    end
end
assert(min(measure_cols(:)) > 0, 'Cannot find some data...');

% getting TensorBoard data (test run)
xtest = cell(ne,1);
ytest = cell(ne,length(tensorboard_measures));
for i = 1:ne
    xtest{i} = CSVtest{i}(:,2);
    for j = 1:length(tensorboard_measures)
        ytest{i,j} = CSVtest{i}(:,measure_cols(i,j));
    end
end

% plotting
for j = 1:length(tensorboard_measures)
    if fixg && (j == 4 || j == 8)
        if j == 4
            [xm,ym] = prepare_data(x,y,2);
        else
            [xm,ym] = prepare_data(x,y,7);
        end
        ym2 = ym;
        [xm,ym] = prepare_data(x,y,j);
        ym(1,:) = ym2(1,:);
    else
        [xm,ym] = prepare_data(x,y,j);
    end
    if strcmp(measure_labels{j}, 'Blinks')
        xm = xm * density_window;
    end    
    plotfig(xm,ym,explabels,measure_labels{j},figdest,measure_file_labels{j}, color_pattern, no_y_label);
end

if fixg
    ym = zeros(size(ym(1,:)));
    t = 1;
    while true
        ym(229*t:228*(t+1)) = 1;
        t = t + 3;
        if 228*(t+1) > length(ym)
            break;
        end
    end
    plotfig(xm,ym,{'video signal'},{'greenish portion'},figdest,'greenish', 1, no_y_label);
end

% some output
disp('MI Full (test data, last step):');
for i = 1:ne
    disp(['   ' explabels{i} ': ' num2str(ytest{i,1}(end))]);    
end

function plotfig(x, y, exp_labels, y_label, fig_dest, filename, color_pattern, no_y_label)
    styles = {'-', '--', ':', '-.'};
    widths = {3, 3, 3, 3};
    if color_pattern == 1
        colors = {[0.749019622802734 0 0.749019622802734], [0.87058824300766 0.490196079015732 0], [0 0 1], [0.501960813999176 0.501960813999176 0.501960813999176]};
    else
        if color_pattern == 2
            colors = {[0 0.498039215803146 0], [1 0 0], [0 0 1]};
        else
            if color_pattern == 3
                colors = {[0 0.447058826684952 0.74117648601532], [0.749019622802734 0 0.749019622802734], [1 0 0]};
            else
                colors = {[0.929411768913269 0.694117665290833 0.125490203499794], [0.635294139385223 0.0784313753247261 0.184313729405403]};
            end
        end
    end
    f = figure;
    ax = axes(f,'FontSize',22,'Box','on');
    hold all;
    ne = size(y,1);
    
    for i = 1:ne
        plot(x,y(i,:), 'DisplayName', exp_labels{i}, 'LineWidth', widths{mod(i-1,length(widths))+1}, 'LineStyle', styles{mod(i-1,length(styles))+1}, 'Color', colors{mod(i-1,length(colors))+1});
    end
    xlim([min(x),max(x)]);
    ylim([min(y(:))*0.95, max(y(:))*1.05]);
    xlabel('Frame'); %,'Interpreter','latex');
    if no_y_label == false
        ylabel(y_label); %,'Interpreter','latex');
    end
    legend('show', 'Location', 'best');
    
    outerpos = ax.OuterPosition;
    ti = ax.TightInset; 
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    fixFactor = 0.98;
    ax.Position = [left bottom ax_width*fixFactor ax_height*fixFactor];
    
    f.PaperPositionMode = 'auto';
    fig_pos = f.PaperPosition;
    f.PaperSize = [fig_pos(3) fig_pos(4)];

    saveas(f, [fig_dest '/' filename '.fig']);
    saveas(f, [fig_dest '/' filename '.pdf']);
end

function [xm,ym] = prepare_data(x, y, measure)
    T = length(y{1,measure});
    ne = size(y,1);
    for i = 1:ne
        T = min(T,length(y{i,measure}));
    end
    xm = zeros(ne,T);
    ym = zeros(ne,T);
    for i = 1:ne
        ym(i,:) = y{i,measure}(1:T);
        if i == 1
            xm = x{i}(1:T);
        end
    end
end

function yp = to_density(y,w)
    yp = zeros(max(length(y),2),1);
    ma = yp(1);
    if w < 0.0
        %gaussianf = gausswin(-w);    
        sigma = w; 
        a = (1:floor(w/2)); 
        b = exp(-(a.^2)/(2*sigma*sigma));
        gaussianf = [b,1.0,b];
        gaussianf = gaussianf ./ sum(gaussianf);
        yp = filter(gaussianf,1,y);    
        return
    end
    k = 1;
    for i = 1:w:length(y)
        if w <= 1
            ma = (1.0-w)*ma + w*y(i);
            yp(i) = ma;
        else
            if i + w - 1 <= length(y)
                s = double(sum(y(i:i+w-1)));
                d = s / w;
            else
                if i == 1
                    w = length(y);
                    s = double(sum(y(i:i+w-1)));
                    d = s / w; 
                    yp(1) = d;
                    yp(2) = d;
                    yp = yp(1:2);
                else
                    if i == 2
                        yp(2) = yp(1);
                        yp = yp(1:2);
                    else
                        yp = yp(1:i-1);
                    end
                end
                break
            end 
            
            yp(k) = d;
            k = k + 1;
        end
    end
    
    yp = yp(1:k-1);
end


