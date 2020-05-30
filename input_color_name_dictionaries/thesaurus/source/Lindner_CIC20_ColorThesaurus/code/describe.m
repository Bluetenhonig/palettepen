%
% Copyright (c) 2012 Albrecht Lindner (ajl.epfl@gmail.com)
% All rights reserved
%
% License: Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)
% 
% If you are using (parts of) this code, please cite the corresponding publication:
% Albrecht Lindner, Bryan Zhi Li, Nicolas Bonnier, and Sabine S?sstrunk, A large-scale multi-lingual color thesaurus, IS&T Color and Imaging Conference, 2012.

function varargout = describe(imagePath, varargin)
% [desc1, desc2, ...] = mydescribe(imagePath, descName1, descName2, ...)
% 
% computes descriptors (descName1, descName2, ...) for image given by imagePath.
%
% examples: desc = mydescribe('./myimage.jpg', 'lab_hist15');
%           [desc_lab, desc_ab] = mydescribe('./myimage.jpg', 'lab_hist15', 'ab_hist21');


%% pre-processing
init;
argn = 1;

% keyboard

ALL = 0;
FORCE = opts.FORCE || opts.FORCE_DESCRIBE;
FORCE = 1;

LD = size(varargin,2);
if (LD == 0)
    ALL = 1;
end

% descriptors the calling function wants to compute
wishlist = '';
for i = 1:LD
    wishlist = [wishlist '<' varargin{i} '>'];
end

% path of mat file where the descriptors are located on the hd
dpath = strrep(imagePath, 'images', 'descriptors');
dpath = [dpath '.mat'];
fprintf('%s ... ', dpath);

% create mat file if not existant
mkpath(dpath);
%origPath = pwd;
%[ddir,file,ext] = fileparts(dpath);
%fileName = strcat(file,ext);
if ~exist(dpath, 'file')
    save(dpath, 'imagePath');
end

warning('off', 'MATLAB:load:variableNotFound');
% keyboard
%% descriptor computations
% you can add as many descriptors as you want by adding a new if statement.
% 
% try to load the descriptor first, if not existant compute it and save it
% for later use
descName = 'lab_hist15_100';
if ~isempty(strfind(wishlist, ['<' descName '>'])) || ALL
    wishlist = strrep(wishlist, ['<' descName '>'], '');
    fprintf('%s ... ', descName);
    load(dpath, descName);
    if ~exist(descName, 'var') || FORCE
        if ~exist('lab', 'var')
            if ~exist('rgb', 'var')
                srgb = im2double(read(imagePath));
%                 keyboard
            end
            lab = mexsRGB2Lab(srgb);
        end
        
        [H, W, D] = size(lab);
        lab_ = reshape(lab, H*W, D);
        lab_hist15 = histnd(double(lab_), [[0 15 100]; [-100 15 100]; [-100 15 100]]);
        lab_hist15 = lab_hist15 / sum(lab_hist15(:));
        lab_hist15 = lab_hist15(:);
        
        save(dpath, descName, '-append');
    end
    varargout{argn} = lab_hist15;
    argn = argn + 1;
end

descName = 'lab_hist15_80';
if ~isempty(strfind(wishlist, ['<' descName '>'])) || ALL
    wishlist = strrep(wishlist, ['<' descName '>'], '');
    fprintf('%s ... ', descName);
    load(dpath, descName);
    if ~exist(descName, 'var') || FORCE
        if ~exist('lab', 'var')
            if ~exist('rgb', 'var')
                srgb = im2double(read(imagePath));
%                 keyboard
            end
            lab = mexsRGB2Lab(srgb);
        end
        
        [H, W, D] = size(lab);
        lab_ = reshape(lab, H*W, D);
        lab_hist15_80 = histnd(double(lab_), [[0 15 100]; [-80 15 80]; [-80 15 80]]);
        lab_hist15_80 = lab_hist15_80 / sum(lab_hist15_80(:));
        lab_hist15_80 = lab_hist15_80(:);
        
        save(dpath, descName, '-append');
    end
    varargout{argn} = lab_hist15_80;
    argn = argn + 1;
end


if length(wishlist) >= 1
    error('unsupprted descriptor types: %s\n', wishlist);
end

fprintf('\n');
warning('on', 'MATLAB:load:variableNotFound');

function img = read(path)
[img, colmap] = imread(path);

% check for indexed image and fix
if ~isempty(colmap)
    tmp = zeros(numel(img), 3);
    tmp(:,1) = colmap(img+1, 1);
    tmp(:,2) = colmap(img+1, 2);
    tmp(:,3) = colmap(img+1, 3);
    img = reshape(tmp, [size(img) 3]);
end


