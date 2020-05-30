%
% Copyright (c) 2012 Albrecht Lindner (ajl.epfl@gmail.com)
% All rights reserved
%
% License: Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)
% 
% If you are using (parts of) this code, please cite the corresponding publication:
% Albrecht Lindner, Bryan Zhi Li, Nicolas Bonnier, and Sabine S?sstrunk, A large-scale multi-lingual color thesaurus, IS&T Color and Imaging Conference, 2012.

function [Lab0s, RGB0s, LabBils, RGBBils] = ColorValues(LANG)

init;

descName = 'lab_hist15_80';
dDim = [3375 1];
N = 15;
cNames = opts.cNames;
Nc = length(cNames);

Lab0s = zeros([Nc 3]); % Lab values of maximum z value
RGB0s = zeros([Nc 3]); % RGB values of maximum z value
LabBils = zeros([Nc 3]); % Lab values after interpolation
RGBBils = zeros([Nc 3]); % RGB values after interpolation

for c = 1:Nc
    cName = cNames{c};
    
    clear('Lidx', 'aidx', 'bidx', 'RGB0', 'Lab0', 'RGBBil', 'LabBil');
    fname = sprintf(opts.fitPath, LANG, cName, descName);
    fprintf('%d: %s', c, fname);
    if exist(fname, 'file')% && ~FORCE
%        keyboard
        load(fname, 'Lidx', 'aidx', 'bidx', 'RGB0', 'Lab0', 'RGBBil', 'LabBil');
        fprintf(' ... loaded');
    end
%     keyboard
    if ~exist('LabBil', 'var') % compute it
        fprintf(' ... computing:');
        % load z values
        zfile = sprintf(opts.zPath, LANG, cName, descName);    
        if exist(zfile, 'file')
           load(zfile, 'zvalues');
        else
            fprintf(' ColorFit: cannot find %s\n', zfile);
        end
        if sum(isnan(zvalues(:))) == prod(dDim)
            fprintf('NO DATA\n');
            
            Lidx = nan;
            aidx = nan;
            bidx = nan;
            
            RGB0 = nan(3,1);
            Lab0 = nan(3,1);
            
            RGBBil = nan(3,1);
            LabBil = nan(3,1);
            
            mkpath(fname);
            save(fname, 'Lidx', 'aidx', 'bidx', 'RGB0', 'Lab0', 'RGBBil', 'LabBil');
            continue
        end
        
        %% find maximum bin and get Lab and RGB coordinates
        fprintf(' max bin');
        % bin centers
        L = 100*linspace(1/2/N, 1-1/2/N, N);
        a = 160*linspace(1/2/N, 1-1/2/N, N)-80;
        b = 160*linspace(1/2/N, 1-1/2/N, N)-80;
        
        % max and indexes for L, a, and b arrays
        [A0 idx] = max(zvalues);
        
        bidx = floor((idx-1) / N^2);
        aidx = idx - bidx*N^2;
        aidx = floor((aidx-1) / N);
        Lidx = idx - bidx*N^2 - aidx*N;
        
        aidx = aidx + 1;
        bidx = bidx + 1;
        
        % color values at maximum bin center
        Lab0 = [L(Lidx) a(aidx) b(bidx)];
        RGB0 = round(mexLab2sRGB(Lab0)*255);
        
        %% bilinear interpolation of center
        fprintf(', bilinear');
        [~, s_lab]  = samples(descName);
        s_lab = reshape(s_lab, [N N N 3]);

        z = double(reshape(zvalues, [N N N]));
        
        isNotBorder_L = ~(Lidx==1 || Lidx == N);
        isNotBorder_a = ~(aidx==1 || aidx == N);
        isNotBorder_b = ~(bidx==1 || bidx == N);

        sum_z = 0;
        sum_L = 0;
        sum_a = 0;
        sum_b = 0;

        for iL = Lidx-isNotBorder_L:Lidx+isNotBorder_L
            for ia = aidx-isNotBorder_a:aidx+isNotBorder_a
                for ib = bidx-isNotBorder_b:bidx+isNotBorder_b
                    curr_z = z(iL, ia, ib);
                    if abs(curr_z) ~= Inf && ~isnan(curr_z)
                        sum_z = sum_z + curr_z;

                        sum_L = sum_L + curr_z*s_lab(iL, ia, ib, 1);
                        sum_a = sum_a + curr_z*s_lab(iL, ia, ib, 2);
                        sum_b = sum_b + curr_z*s_lab(iL, ia, ib, 3);
                    end
                end
            end
        end
        LabBil = [sum_L sum_a sum_b] / sum_z;
        RGBBil = round(mexLab2sRGB(LabBil)*255);
        
        %% save stuff
        mkpath(fname);
        save(fname, 'Lidx', 'aidx', 'bidx', 'RGB0', 'Lab0', 'RGBBil', 'LabBil');
    end
    Lab0s(c, :) = reshape(Lab0, [1 3]);
    RGB0s(c, :) = reshape(RGB0, [1 3]);
    LabBils(c, :) = reshape(LabBil, [1 3]);
    RGBBils(c, :) = reshape(RGBBil, [1 3]);
    fprintf('\n');
end

figure(1)
imagesc(reshape(RGBBils/255, [1 Nc 3]));
set(gca, 'XTick', 1:Nc, 'XTickLabel', opts.cNames, 'YTick', []);

