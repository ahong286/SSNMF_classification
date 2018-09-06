function [datapatches] = ExtractPatchesByIndexGabor(bandnum, Index,patchsize,datasetname )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
padx = fix(patchsize(1)/2);
pady = fix(patchsize(2)/2);
offsetx = padx * 2;
offsety = pady * 2;
feaNum = numel(Index);

pdim = patchsize(1) * patchsize(2);
datapatches = cell(bandnum,1);
% gaborNum = 40;

for k=1:bandnum
    sfile = sprintf('gaborfea_%d.mat',k);
    load(fullfile('gaborfea',datasetname,sfile), 'gaborfea');
    [m, n, p] = size(gaborfea);
    feaDim = pdim * p;
    feas = zeros(feaDim,feaNum);
    for pp=1:p
        tmpFea=zeros(pdim,feaNum);
        padData = padarray(gaborfea(:,:,pp),[padx,pady],'replicate','both');
        for i=1:feaNum
            [x,y] = ind2sub([m,n],Index(i));
            tmpFea(:,i) = reshape(padData(x:x+offsetx,y:y+offsety),pdim,1); 
        end
        feas(((pp-1)*pdim+1):(pp*pdim),:) = tmpFea;
    end
%     Tr = NormalizeFea(Tr,0);
    datapatches{k}=feas;   
    clear tmpFea feas;
end
clear padData;
end

