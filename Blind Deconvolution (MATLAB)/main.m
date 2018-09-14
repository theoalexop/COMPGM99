clc; clear all; close all;

% For the purpose of loading NIfTI files in MATLAB [1], the following tool 
% is utilized: 
% J. Shen (2014). "Tools for NIfTI and ANALYZE image." 
% MATLAB Central File Exchange. 
% Online: https://uk.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
% Retrieved: June 5, 2018.
% Furthermore, MATLAB provides blind deconvolution as a built-in function [2].
% [1] MATLAB Version 8.5 (Release 2015a), MathWorks, Natick, MA, USA, 2015.
% [2] https://www.mathworks.com/help/images/ref/deconvblind.html

% A blurred and noisy image from the HH dataset is loaded, along with the
% true PSF which is saved in .mat format in Python.
input_img = load_untouch_nii('IXI080-HH-1341-T1-blurred-noisy.nii.gz');
load('PSFi.mat');

% The aforementioned image is deconvolved based on blind deconvolution
% utilizing the true PSF.
[deconv_img, PSFr] = deconvblind(input_img.img, PSFi, 4);

% The deconvolved image is saved in .mat format for futher visualization in
% Python.
save('deconv.mat','deconv_img');