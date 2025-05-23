# Fluorescence-tomography-Recon

This is a project of Bessel Beam flourence tomography simulation and reconstruction, and is based on the paper in Cell: https://www.cell.com/cell/fulltext/S0092-8674(23)00412-9. The code simulates how the beam illuminates and scans the sample in multiple different angles on a single volume, and the volume is then reconstructed using these multiple projections.

The code is divided into 2 parts, the original unmodified code from the paper exists in the folder *2pSAM_recon-main*, and the modified version is inside the folder *2pSAM_recon-main-revised*. 

This readme is a rough break down of the code structure inside *2pSAM_recon-main-revised*.

**Note: Aside from additions on functionality, the modified version also removes the gpu built-in functions in Matlab in the original code, because they can be only ran when there exists a gpu, else it would error and crash the program.**

Directly inside *2pSAM_recon-main-revised*, there are several code files that can be ran, and initiates processes.

## run_PSFs.m:

This file generates PSF files and save them inside the results folder. This file is inherited from the original code base. For detailed parameter setting, see code.

## run_plot.m:

This file takes in the PSF files, and outputs plots of different angles, including TIF, fig, and jpg files, and automatically saves them. 
This file also calculates the half-max, but does not automatically saves the output figure, because there is one interval variable that needs manual tuning.

This file uses a helper function that are also present here:
*displaySectionDataByImage.m*

Note: *imageshow.m* is a rudimentary visualization function that is now not used. 

## run_Generate.m:

This file takes in a 3D volume data and the generated PSF, and outputs the projections file to simulate the output of the scanning process.

## generatingCubicBodyWithBalls.m:

This file generates random balls inside a set volume for simulation and recon testing.

## run_recon.m:

This file takes in the projections with the PSF, and outputs a reconstructions 3d volume. This file is inherited from the original code base. For detailed parameter settings, see code.

## napari_view.py

This file is a simple example that takes in the paths of several TIF files, and visualizes them in napari.

Directly inside *2pSAM_recon-main-revised*, the rest of the code files are three different regularizer functions, which are *regula.m*, *regulaRoot.m*, and *regulaTV.m*. *regulaRootHelper* is a helper function function of *regulaRoot.m*.

The **util** subfolder are all inherited from the original code base. Inside it, the important subfolders are **utils_PSF** and **utils_recon**


