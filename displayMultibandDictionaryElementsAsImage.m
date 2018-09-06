function I = displayMultibandDictionaryElementsAsImage(D, numRows, numCols,X,Y,sortVarFlag)
% function I = displayDictionaryElementsAsImage(D, numRows, numCols, X,Y)
% displays the dictionary atoms as blocks. For activation, the dictionary D
% should be given, as also the number of rows (numRows) and columns
% (numCols) for the atoms to be displayed. X and Y are the dimensions of
% each atom.
%  numRows       --the band numbers
%  numCols       --the dictionary numbers of every band
%  X             --the row numbers of patch
%  Y             --the col numbers of patch

borderSize = 1;
columnScanFlag = 1;
strechEachVecFlag = 1;
showImFlag = 1;

if (length(who('X'))==0)
    X = 8;
    Y = 8;
end

if (length(who('showImFlag'))==0) 
    showImFlag = 1;
end

%%% construct the image to display (I)
sizeForEachImage = floor(sqrt(size(D{1},1))+borderSize);

I = zeros((sizeForEachImage*numRows+borderSize),(sizeForEachImage*numCols+borderSize),3);
%%% fill all this image in blue
I(:,:,1) = 0;%min(min(D));
I(:,:,2) = 0; %min(min(D));
I(:,:,3) = 1; %max(max(D));

for j = 1:numRows
    C = D{j};
    for counter = 1:size(C,2)
        C(:,counter) = C(:,counter)-min(C(:,counter));
        if (max(C(:,counter)))
            C(:,counter) =C(:,counter)./max(C(:,counter));
        end
    end
    for i = 1:numCols
%         if (strechEachVecFlag)
%             D(:,counter) = D(:,counter)-min(D(:,counter));
%             D(:,counter) = D(:,counter)./max(D(:,counter));
%         end
%         if (columnScanFlag==1)
%             I(borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,1)=reshape(D(:,counter),8,8);
%             I(borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,2)=reshape(D(:,counter),8,8);
%             I(borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,3)=reshape(D(:,counter),8,8);
%         else
            % Go in Column Scan:
            I(borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,1)=reshape(C(:,i),X,Y);
            I(borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,2)=reshape(C(:,i),X,Y);
            I(borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,3)=reshape(C(:,i),X,Y);
%         end
    end
end
%  figure; imshow(I(1001:1501,:,:))
if (showImFlag) 
%     I = I-min(min(min(I)));
%     I = I./max(max(max(I)));
    figure; imagesc(I(501:1001,:,:))
%      figure; imagesc(I(1:300,:,:));
%    figure; imshow(I(1:300,:));
%     figure;imagesc(I(301:end,:,:));
 %    figure;imshow(I(301:end,:));
%     imshow(I,[]);
end
