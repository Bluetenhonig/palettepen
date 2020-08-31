%
% Copyright (c) 2012 Albrecht Lindner (ajl.epfl@gmail.com)
% All rights reserved
%
% License: Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)
% 
% If you are using (parts of) this code, please cite the corresponding publication:
% Albrecht Lindner, Bryan Zhi Li, Nicolas Bonnier, and Sabine S?sstrunk, A large-scale multi-lingual color thesaurus, IS&T Color and Imaging Conference, 2012.

function [srgb, lab]  = samples(type)

switch type
    case {'lab_hist15'}
        l = 100*linspace(1/30, 1-1/30, 15)';
        l = repmat(l, [1 15 15]);
        a = 200*linspace(1/30, 1-1/30, 15)-100;
        a = repmat(a, [15 1 15]);
        b = reshape(200*linspace(1/30, 1-1/30, 15)-100, [1 1 15]);
        b = repmat(b, [15 15 1]);
        
        lab = [l(:), a(:), b(:)];
        srgb = mexLab2sRGB(lab);
    
    case {'lab_hist15_80'}
        l = 100*linspace(1/30, 1-1/30, 15)';
        l = repmat(l, [1 15 15]);
        a = 160*linspace(1/30, 1-1/30, 15)-80;
        a = repmat(a, [15 1 15]);
        b = reshape(160*linspace(1/30, 1-1/30, 15)-80, [1 1 15]);
        b = repmat(b, [15 15 1]);
        
        lab = [l(:), a(:), b(:)];
        srgb = mexLab2sRGB(lab);
end
