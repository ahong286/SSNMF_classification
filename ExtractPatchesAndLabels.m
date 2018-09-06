function [datapatches, labels] = ExtractPatchesAndLabels( datacube,groundtruth,patchsize )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
padx = fix(patchsize(1)/2);
pady = fix(patchsize(2)/2);
offsetx = padx * 2;
offsety = pady * 2;
[m, n, b] = size(datacube);
pdim = patchsize(1) * patchsize(2);
datapatches = cell(b,1);
datacubeP = padarray(datacube,[padx,pady],'replicate','both');
labels = zeros(m*n,3);
count = 0;
for raw=1:m
    for col=1:n
        count = count +1;
        labels(count,1) = groundtruth(raw,col);
        labels(count,2) = raw;
        labels(count,3) = col;
        for k=1:b
            A = reshape(datacubeP(raw:raw+offsetx, col:col+offsety, k), pdim, 1);
            datapatches{k} = [datapatches{k}, A];
            A = [];
        end
    end
end
clear DataCubeB;
end

