function [datapatches] = ExtractByIndexGabor(bandnum, Index,datasetname )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
feaNum = numel(Index);
datapatches = cell(bandnum,1);
% gaborNum = 40;

for k=1:bandnum
    sfile = sprintf('gaborfea_%d.mat',k);
    load(fullfile('gaborfea',datasetname,sfile), 'gaborfea');
    [m, n, p] = size(gaborfea);
    feas = zeros(p,feaNum);
    for i=1:feaNum
        [x,y] = ind2sub([m,n],Index(i));
        feas(:,i) = gaborfea(x,y,:); 
    end
    datapatches{k}=feas;   
    clear tmpFea feas;
end
clear padData;
end

