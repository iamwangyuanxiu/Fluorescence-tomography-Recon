from skimage.io import imread
import napari
viewer = napari.Viewer()
# path = 'data//figures//Bessel_0.35_NA_0.8_annulus_right_pixels//'
# image = []
# for i in range(1, 14):
#     path_temp = path + 'Beam'+ str(14-i) + '.tif'
#     print(path_temp)
#     image.append(imread(path_temp))
#     viewer.add_image(image[i-1])
    

path = 'data//images//3d_400um_to_500um_cropped_xp4um.tif'
path1 = 'data//sim_results//400_xy_pixel_dz_1//Bessel_0.35NA_0.8_annulus//recon_RL_zoom6ca128up1_-25-1-25_pupNO_en0_K0bias0_DAO1_EW0.5_mI30_mC0_frames1-13-13//frame_1.tif'
path2 = 'data//sim_results//default//Bessel_0.2NA_0.7_annulus_reg0_1//recon_RL_zoom6ca128up1_-25-1-25_pupNO_en0_K0bias0_DAO1_EW0.5_mI10_mC0_frames1-13-13//frame_1.tif'
# """ path = 'data//sim_results//Bessel_0.2NA_0.7_annulus_25x_reg0_balls1//recon_RL_zoom4ca512up1_-30-0.5-30_pupNO_en0_K0bias0_DAO1_EW0.5_mI10_mC0_frames1-13-13//frame_1.tif'
# path1 = 'data//images//balls//balls.tif'
# path2 = 'data//sim_results//2pSAM_25x_balls1//recon_RL_zoom4ca512up1_-30-0.5-30_pupNO_en0_K0bias0_DAO1_EW0.5_mI10_mC0_frames1-13-13//frame_1.tif' """

# path2 = 'data//sim_results//Bessel_0.25NA_0.4_annulus_332_1//recon_RL_zoom6.4863ca332up1_-25-1-25_pupNO_en0_K0bias0_DAO1_EW0.5_mI30_mC0_frames1-13-13//frame_1.tif'
# path3 = 'data//sim_results//Bessel_0.25NA_0.2_annulus_332_1//recon_RL_zoom6.4863ca332up1_-25-1-25_pupNO_en0_K0bias0_DAO1_EW0.5_mI30_mC0_frames1-13-13//frame_1.tif'
# path4 = 'data//sim_results//Bessel_0.25NA_0.1_annulus_332_1//recon_RL_zoom6.4863ca332up1_-25-1-25_pupNO_en0_K0bias0_DAO1_EW0.5_mI30_mC0_frames1-13-13//frame_1.tif'

# path5 = 'data//sim_results//2pSAM_332_1//recon_RL_zoom6.4863ca332up1_-25-1-25_pupNO_en0_K0bias0_DAO1_EW0.5_mI30_mC0_frames1-13-13//frame_1.tif'

original = imread(path)
Bessel02007 = imread(path2)
Bessel03508_332 = imread(path1)
# # Bessel02504 = imread(path2)
# # Bessel02502 = imread(path3)
# # Bessel02501 = imread(path4)
# two_psam = imread(path5)
stripes = imread(path2)
# #two_psam = imread(path2)
# #viewer.add_image(original)
# viewer.add_image(Bessel02507)
# # viewer.add_image(Bessel02504)
# # viewer.add_image(Bessel02502)
# # viewer.add_image(Bessel02501)
# #viewer.add_image(stripes)
viewer.add_image(original)
viewer.add_image(Bessel02007)
viewer.add_image(Bessel03508_332)
#viewer.add_image(two_psam)

contrast = (0, 1500)
viewer.layers['original'].contrast_limits = contrast
viewer.layers['Bessel02007'].contrast_limits = contrast
viewer.layers['Bessel03508_332'].contrast_limits = contrast
# viewer.layers['two_psam'].contrast_limits = contrast
# viewer.layers['Bessel02507'].contrast_limits = contrast

napari.run()