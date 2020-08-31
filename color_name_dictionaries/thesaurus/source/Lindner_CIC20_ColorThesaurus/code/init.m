%
% Copyright (c) 2012 Albrecht Lindner (ajl.epfl@gmail.com)
% All rights reserved
%
% License: Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)
% 
% If you are using (parts of) this code, please cite the corresponding publication:
% Albrecht Lindner, Bryan Zhi Li, Nicolas Bonnier, and Sabine S?sstrunk, A large-scale multi-lingual color thesaurus, IS&T Color and Imaging Conference, 2012.

clear opts;

opts.FORCE = 1;
opts.FORCE_COLLECT = 0;
opts.FORCE_DESCRIBE = 0;
opts.FORCE_Z = 0;


opts.db = './DB/';

opts.imagePath = [opts.db 'lang_%s/images/%s/%s'];

opts.descPath = [opts.db 'lang_%s/descriptors/%s/%s.mat'];

opts.collPath = [opts.db 'lang_%s/collections/%s/%s.mat'];

opts.zPath = [opts.db 'lang_%s/z/%s/%s.mat'];

opts.fitPath = [opts.db 'lang_%s/fit/%s/%s.mat'];

% add more color names if necessary
opts.cNames = {'white' 'black' 'red' 'green' 'yellow' 'blue' 'brown' 'purple' 'pink' 'orange' 'grey', 'ultramarine'};
opts.Nc = length(opts.cNames);

% add more languages if necessary
opts.languages = {'en'};
