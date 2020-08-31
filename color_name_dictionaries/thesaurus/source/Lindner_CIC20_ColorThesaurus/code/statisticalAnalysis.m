%
% Copyright (c) 2012 Albrecht Lindner (ajl.epfl@gmail.com)
% All rights reserved
%
% License: Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)
% 
% If you are using (parts of) this code, please cite the corresponding publication:
% Albrecht Lindner, Bryan Zhi Li, Nicolas Bonnier, and Sabine S?sstrunk, A large-scale multi-lingual color thesaurus, IS&T Color and Imaging Conference, 2012.

function zmatrix = statisticalAnalysis(LANG)

% Change the character encoding due to Windows or Mac
% slCharacterEncoding('ISO-8859-1');

init;
FORCE = opts.FORCE || opts.FORCE_Z;
% FORCE = 1;
% keyboard

descName = 'lab_hist15_80';
dDim = [3375 1];
% descName = 'ab_hist21';
% descName = 'lab_hist9';
cNames = opts.cNames;
% cNames = {'reddish purple', 'darkgreen', 'red', 'blue'};
Nc = length(cNames);
% keyboard
% fname = sprintf(opts.compPath, desctype);
zmatrix = cell(Nc, 1);
% Nclasses = zeros(Nc, 1);

for c = 1:Nc
%     tic
    cName = cNames{c};

    fname = sprintf(opts.zPath, LANG, cName, descName);
    fprintf('%s', fname);

    warning('off', 'MATLAB:load:variableNotFound');
    if exist(fname, 'file') && ~FORCE
       load(fname, 'zvalues');
    end
    
    if exist('zvalues', 'var') && ~FORCE
       fprintf(' ... loaded\n');
    else % cannot load from file, have to compute distances
        fprintf(' ... computing\n');
        if ~exist('desc_all', 'var');
            desc_all = cell(30, Nc);
            for i = 1:Nc
                tmp = collect(cNames{i}, LANG, descName);
                desc_all(1:length(tmp), i) = tmp;
            end
%             keyboard
            fprintf('compute ranks ... ');
            Tstart = tic;
            ss = size(desc_all);
            valid = true(ss);
            for i2 = 1:ss(2)
                for i1 = ss(1):-1:1
                    if isempty(desc_all{i1, i2})
                        valid(i1, i2) = 0;
                        desc_all{i1, i2} = single(Inf(dDim));
                    else
                        break
                    end
                end
            end
            
            desc_all = reshape(desc_all, [1 ss]);
            desc_all = cell2mat(desc_all);
            ss = size(desc_all);
            rank_all = reshape(desc_all, [ss(1) ss(2)*ss(3)]);
            rank_all = mexranks(rank_all')';
            rank_all = reshape(rank_all, ss);
            Ntot = sum(valid(:));
            fprintf('%f', toc(Tstart));
%             keyboard
        end
        
        Nkw = sum(valid(:, c));
        Nkwn = Ntot - Nkw;
        T = rank_all(:, 1:Nkw, c);
        T = sum(T, 2);
        mu_T = Nkw*(Nkw+Nkwn+1)/2;
        sigma_T = sqrt(Nkw*Nkwn*(Nkw+Nkwn+1)/12);
        zvalues = (T - mu_T) / sigma_T;
        zvalues = zvalues';
        
        mkpath(fname);
        save(fname, 'zvalues');

%         toc
    end
    zmatrix{c} = zvalues;
    clear('zvalues');
end










% function zmatrix = computeZ(LANG)
% 
% % Change the character encoding due to Windows or Mac
% slCharacterEncoding('ISO-8859-1');
% 
% init;
% FORCE = opts.FORCE || opts.FORCE_Z;
% % FORCE = 1;
% % keyboard
% 
% descName = 'lab_hist15_80';
% % descName = 'ab_hist21';
% % descName = 'lab_hist9';
% cNames = opts.cNames;
% % cNames = {'darkgreen'};
% Nc = opts.Nc;
% 
% % fname = sprintf(opts.compPath, desctype);
% zmatrix = cell(Nc, 1);
% Nclasses = zeros(Nc, 1);
% 
% for c = 1:Nc
%     tic
%     cName = cNames{c};
% 
%     fname = sprintf(opts.zPath, LANG, cName, descName);
%     fprintf('%s', fname);
% 
%     warning('off', 'MATLAB:load:variableNotFound');
%     if exist(fname, 'file') && ~FORCE
%        load(fname, 'zvalues', 'Nclass');
%     end
% 
%     if exist('zvalues', 'var') && exist('Nclass', 'var') && ~FORCE
%        fprintf(' ... loaded\n');
%     else % cannot load from file, have to compute distances
%         fprintf(' ... computing\n');
%         if ~exist('desc_all', 'var');
%             desc_all = cell(30, opts.Nc);
%             for i = 1:opts.Nc
%                 tmp = collect(opts.cNames{i}, LANG, descName);
%                 desc_all(1:length(tmp), i) = tmp;
%             end
%         end
%         
%         desc1 = desc_all(:, c)';
%         Nclass = length(desc1);
%         for i = 1:length(desc1)
%             if isempty(desc1{i})
%                 Nclass = i-1;
%                 break;
%             end
%         end
%         desc1 = cell2mat(desc1(1:Nclass));
%         
%         desc2 = desc_all(:, [1:(c-1) (c+1):Nc]);
%         empty = zeros(size(desc2));
%         for i = 1:numel(desc2)
%             empty(i) = isempty(desc2{i});
%         end
%         desc2 = desc2(~empty)';
%         desc2 = cell2mat(desc2);
%         
%         Nbins = size(desc1, 1);
%         zvalues = zeros(1,Nbins);
%         for i = 1:Nbins
%             % compare images with a keyword against images without a keyword
%             [~, ~, stat] = ranksum(desc1(i, :), desc2(i, :));
%             zvalues(i) = stat.zval;
%         end
% 
%         mkpath(fname);
%         save(fname, 'zvalues', 'Nclass');
% 
%         toc
%     end
%     Nclasses(c) = Nclass;
%     zmatrix{c} = zvalues;
%     clear('zvalues');
% end
