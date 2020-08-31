%
% Copyright (c) 2012 Albrecht Lindner (ajl.epfl@gmail.com)
% All rights reserved
%
% License: Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)
% 
% If you are using (parts of) this code, please cite the corresponding publication:
% Albrecht Lindner, Bryan Zhi Li, Nicolas Bonnier, and Sabine S?sstrunk, A large-scale multi-lingual color thesaurus, IS&T Color and Imaging Conference, 2012.

function [descs, Nclass] = collect(cName, LANG, varargin)
init;

FORCE = opts.FORCE || opts.FORCE_COLLECT;
% keyboard
% FORCE = 1;

% parse varargin
descName = 'lab_hist15_80';
% descName = 'ab_hist21';
% descName = 'lab_hist9';
if size(varargin, 2) > 0
    descName = varargin{1};
end

fname = sprintf(opts.collPath, LANG, descName, cName);

% keyboard

fprintf('%s ... ', fname);
if exist(fname, 'file') && ~FORCE
	load(fname, 'descs', 'descName', 'imageNames', 'Nclass');
    fprintf('loaded\n');
else
    fprintf('collecting\n');
    
    % get list of files for this cName
    content = dir(sprintf(opts.imagePath, LANG, cName, ''));
    content = content(3:end);
    imageNames = cell(size(content));
    Nclass = 0;
    for i = 1:length(content)
        if isempty(regexp(content(i).name, '^\.', 'once'))
            Nclass = Nclass + 1;
            imageNames{Nclass} = content(i).name;
        end
    end
    imageNames = imageNames(1:Nclass);
    
    % pre allocate
    descs = cell(Nclass, 1);
    
    % get all descriptors
    count = 1;
    for i = 1:Nclass
        fprintf('%d/%d ', i, Nclass);
        imageName = imageNames{i};
        try
            descs{count} = single(describe(sprintf(opts.imagePath, LANG, cName, imageName), descName));
            imageNames{count} = imageNames{i};
            count = count + 1;
        catch
%             keyboard
            fprintf('ERROR, skip file\n');
        end
    end
%     keyboard
    Nclass = count - 1;
    descs = descs(1:Nclass);
    imageNames = imageNames(1:Nclass);
    
    % save to HD
    mkpath(fname);
    save(fname, 'descs', 'descName', 'imageNames', 'Nclass');
end

