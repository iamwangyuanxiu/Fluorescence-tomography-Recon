%% Code to Generate Projections given the 3D olume and PSFs.

clc; clear; close all; addpath(genpath('utils'));
PSFs_Path = 'data//PSFs//default'; %Check if PSF exists before generating new PSF
EI_path = 'data//sim_results//default//Bessel_0.2NA_0.7_annulus_reg0_1';%file path
%data\sim_results\2pSAM_original_results\recon_fRL_zoom4ca512up1_-30-0.5-30_pupNO_en0_K0bias0_DAO1_EW0.5_mI30_mC0_frames1-13-13
EI_name = 'recon_RL_zoom6ca128up1_-25-1-25_pupNO_en0_K0bias0_DAO1_EW0.5_mI10_mC0_frames1-13-13//Xguess_iter7.tif';
%EI_name = '3d_400um_to_500um_cropped_xp4um.tif';%file name
%EI_name = 'balls.tif';%file name
resultSave_path = ['results//generate_',datestr(now, 'YYYYmmDD_HHMMSS')];%Save results here

%data\sim_results\2pSAM_original_results\recon_fRL_zoom4ca512up1_-30-0.5-30_pupNO_en0_K0bias0_DAO1_EW0.5_mI30_mC0_frames1-13-13

disp('Loading PSFs...');
load([PSFs_Path,'//Bessel_0.2NA_0.7_annulus//PSF_mid_zoom6ca128up1psf128_-25-1-25_scanMode2_2-4-8_maxX8.5000Y8.5000biasX0.0Y0.0_pupNO_en0_K0bias0//PSFParameters.mat']);
for i = 1:PSFParameters.angleNum
    load([PSFs_Path,'//Bessel_0.2NA_0.7_annulus//PSF_mid_zoom6ca128up1psf128_-25-1-25_scanMode2_2-4-8_maxX8.5000Y8.5000biasX0.0Y0.0_pupNO_en0_K0bias0//psf_all_',num2str(i),'.mat'],'psf_thisAngle');
    psfs(:,:,:,i) = psf_thisAngle;
end
disp('PSFs loaded');

% disp('Loading PSFs...');
% load([PSFs_Path,'//25X//PSF_cx3cr1//PSFParameters.mat']);
% for i = 1:PSFParameters.angleNum
%     load([PSFs_Path,'//25X//PSF_cx3cr1//psf_all_',num2str(i),'.mat'],'psf_thisAngle');
%     psfs(:,:,:,i) = psf_thisAngle;
% end
% disp('PSFs loaded');

[psf_r, psf_c, psf_s, angleNum]=size(psfs);

%% loading images
disp('Loading raw data...');
projs = single(loadtiff([EI_path,'//',EI_name]));
[projr_r,projr_c,projr_num]=size(projs);

projs = projs(:,:,1:psf_s);
disp('Raw data loaded');

[proj_r,proj_c,proj_num]=size(projs);

Proj_init = zeros(proj_r,proj_c,angleNum);

for angleNow = 1:angleNum
    %largepsf((proj_r-psf_r)/2+1:(proj_r+psf_r)/2,(proj_c-psf_c)/2+1:(proj_c+psf_c)/2,:)=squeeze(psfs(:,:,:,angleNow));
    ProjNow = forwardProj_RL_GPU(squeeze(psfs(:,:,:,angleNow)),projs);
    Proj_init(:,:, angleNow) = ProjNow;
end
stackTempForSave = gather(Proj_init);
saveastiff_overwrite(stackTempForSave,[resultSave_path '//testing.tif'],0,1);

saveastiff_overwrite(projs,[resultSave_path '//data.tif'],0,1);