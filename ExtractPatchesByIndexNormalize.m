function [datapatches] = ExtractPatchesByIndexNormalize(datacube,m_trainingIndex,patchsize )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
padx = fix(patchsize(1)/2);
pady = fix(patchsize(2)/2);
offsetx = padx * 2;
offsety = pady * 2;
[m, n, b] = size(datacube);
pdim = patchsize(1) * patchsize(2);
datapatches = cell(b,1);
for k=1:b
    Tr=[];
    padData = padarray(datacube(:,:,k),[padx,pady],'replicate','both');
    for i=1:numel(m_trainingIndex)
        [x,y] = ind2sub([m,n],m_trainingIndex(i));
        A = reshape(padData(x:x+offsetx,y:y+offsety),pdim,1);        
        Tr=cat(2,Tr,A);
    end
    Tr = NormalizeFea(Tr,0);
    datapatches{k}=Tr;    
end
clear padData;
end

