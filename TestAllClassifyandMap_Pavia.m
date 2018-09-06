
addpath(fullfile(pwd, 'Datasets/Pavia'));
addpath(fullfile(pwd, 'libsvm-3.20/matlab'));
datasetname = 'PaviaU';

isSelect = 0;
is3DDWT = 0;
isUPlus = 0;   % SWSCmu is applied upon sparse coefficients; 
groundTruth = importdata('PaviaU_gt.mat');
vgroundTruth = reshape(groundTruth, [numel(groundTruth),1]);
[m,n]=size(groundTruth);
% class number
numofClass = max(groundTruth(:));   % It is suitable for Indian_pines data;
% PaviaU_SC_CS_patches5_nBasis30_beta0.10_whole.mat
% PaviaU_SC_CS_patches5_nBasis150_beta0.05_whole_gabor.mat
% PaviaU_GSC_CS_patches5_nBasis30_beta0.1_whole.mat
% PaviaU_GSC_CS_patches5_nBasis150_beta0.05_whole_gabor
%% classify based on the sparse codes and denoising results.

sampleRate = 0.09;
filename = 'PaviaU_GSC_CS_patches5_nBasis30_beta0.10_KNN3_whole';
file = sprintf('%s.mat',filename);
load(file,'S','vLabels', 'index');
S = S';
if isSelect == 1
    % Get training samples and test samples;
    trainingSamples = cell(numofClass,1);
    testingSamples = cell(numofClass,1);
    trainingIndex = cell(numofClass,1);
    testingIndex = cell(numofClass,1);
    trainingLabels = cell(numofClass,1);
    testingLabels = cell(numofClass,1);
    numofTest = zeros(numofClass,1);

    for c = 1: numofClass
        cc  = double(c);
        class = find(vLabels == c);
        if isempty(class)
            continue;
        end
        perm = randperm(numel(class)); 
        breakpoint = round(numel(class)*sampleRate);
        trainingIndex{c} = class(perm(1:breakpoint));
        testingIndex{c} = class(perm(breakpoint+1:end));
        trainingSamples{c} = S(trainingIndex{c},:);
        trainingLabels{c} = vLabels(trainingIndex{c},:);
        testingSamples{c} = S(testingIndex{c},:);
        testingLabels{c} = vLabels(testingIndex{c},:);
        numofTest(c) = numel(testingIndex{c});        
    end
    mtrainingData = cell2mat(trainingSamples);
    mtestingData = cell2mat(testingSamples);
    mtrainingLabels = cell2mat(trainingLabels);
    mtestingLabels = cell2mat(testingLabels);
    mtrainingIndex = cell2mat(trainingIndex);
    mtestingIndex = cell2mat(testingIndex); 
    stdIndex = index;
    save('PaviaU_traintestsets.mat','mtrainingIndex','mtestingIndex','stdIndex','vgroundTruth');
    
    resultMap1 = zeros(size(vgroundTruth));
    resultMap1(index(mtestingIndex)) = vgroundTruth(index(mtestingIndex));
    figure(1), imagesc(reshape(resultMap1,[m,n]));
    resultMap2 = zeros(size(vgroundTruth));
    resultMap2(index(mtrainingIndex)) = vgroundTruth(index(mtrainingIndex));
    figure(2), imagesc(reshape(resultMap2,[m,n]));
    
    mtrainingLabels = double(mtrainingLabels);
    mtestingLabels = double(mtestingLabels);  
    %% classify
    [ predicted_label, rr, prob_estimates ] = SVMClassify( mtrainingLabels, mtrainingData, mtestingLabels,mtestingData ); 

    index = double(index);
    resultMap = vgroundTruth;
    resultMap(index(mtestingIndex)) = predicted_label;
    figure(3), imagesc(reshape(resultMap,[m,n]));

    [overall,kappa,average,classindividual] = calcError(mtestingLabels'-1,predicted_label'-1,[1:numofClass]);
    resultsFile = sprintf('results_%s_%s.mat',filename,sampleRate);
    save(resultsFile, 'overall','kappa','average','classindividual','predicted_label','mtestingLabels');   

elseif (is3DDWT == 1)
    % 3DWTF--Yuntao Qian & Jun Zhou, a smooth are execute in dwt3d_feature function, if the filter_mask is 9*9, 
    % the recognition rate will increase.
    
    load('PaviaU_traintestsets.mat','mtrainingIndex','mtestingIndex','stdIndex');

    DataFile = 'PaviaU.mat'
    rawData = importdata(DataFile);% Load hyperspectral image and groud truth
    if ndims(rawData) ~= 3 % save time
        return;
    end
    [m, n, b] = size(rawData); 
    feats = single(dwt3d_feature(rawData));
    vdataCube = reshape(feats,[m*n,15*b]);
    
    mtrainingData = vdataCube(stdIndex(mtrainingIndex),:);
    mtrainingLabels = vgroundTruth(stdIndex(mtrainingIndex),:); 
    mtestingData = vdataCube(stdIndex(mtestingIndex),:);
    mtestingLabels = vgroundTruth(stdIndex(mtestingIndex),:);  

    mtrainingLabels = double(mtrainingLabels);
    mtestingLabels = double(mtestingLabels); 
   mtrainingData = double(mtrainingData);
   mtestingData =  double(mtestingData);
    %% classify
    [ predicted_label, rr, prob_estimates ] = SVMClassify( mtrainingLabels, mtrainingData, mtestingLabels,mtestingData ); 

    resultMap = vgroundTruth;
    resultMap(stdIndex(mtestingIndex)) = predicted_label;
    figure, imagesc(reshape(resultMap,[m,n]));

    [overall,kappa,average,classindividual] = calcError(mtestingLabels'-1,predicted_label'-1,[1:numofClass]);
    resultsFile = sprintf('results_%s_%s_3DDWT.mat',datasetname,sampleRate);
    save(resultsFile, 'overall','kappa','average','classindividual','predicted_label','mtestingLabels');
    
elseif (isUPlus == 1)
    N = size(index,1);
    cord = zeros(N,2);
    for i = 1:N
         [row, col] = ind2sub([m,n],index(i));
         cord(i,:) = [row, col];
    end
    [N, M]=size(S);
    patchwiseMean=zeros(N,M);
    patchwiseStd=zeros(N,M);
    window = 7;
    p=floor(window/2);
    for i=1:N
        inds =  (cord(:,1)>=cord(i,1)-p) & (cord(:,1)<=cord(i,1)+p) & (cord(:,2)<=cord(i,2)+p) & (cord(:,2)>=cord(i,2)-p) ;
        S_Mean(i,:)=mean(S(inds,:),1);
        S_Std(i,:)=std(S(inds,:),0,1);
    end  
    S = S_Mean;
    
    load('PaviaU_traintestsets.mat','mtrainingIndex','mtestingIndex','stdIndex');
    trIndex = [];
    tstIndex = [];
    for i =1:length(mtestingIndex)
        curIndex = mtestingIndex(i);
        tstIndex = [tstIndex, find(index==stdIndex(curIndex))];
    end
    for j=1:length(mtrainingIndex)
        curIndex = mtrainingIndex(j);
        trIndex = [trIndex, find(index==stdIndex(curIndex))];    
    end
    mtrainingData = S(trIndex,:);
    mtrainingLabels = vLabels(trIndex,:); 
    mtestingData = S(tstIndex,:);
    mtestingLabels = vLabels(tstIndex,:);  
    
    mtrainingLabels = double(mtrainingLabels);
    mtestingLabels = double(mtestingLabels);  
    %% classify
    [ predicted_label, rr, prob_estimates ] = SVMClassify( mtrainingLabels, mtrainingData, mtestingLabels,mtestingData ); 

    index = double(index);
    resultMap = vgroundTruth;
    resultMap(index(tstIndex)) = predicted_label;
    figure, imagesc(reshape(resultMap,[m,n]));

    [overall,kappa,average,classindividual] = calcError(mtestingLabels'-1,predicted_label'-1,[1:numofClass]);
    resultsFile = sprintf('results_%s_%0.2f_mu.mat',filename,sampleRate);
    save(resultsFile, 'overall','kappa','average','classindividual','predicted_label','mtestingLabels');      
    
else
    % Compute the recognition rate of different methods on above training samples and test samples;
    % Firstly get corresponding the training data and test data from the
    % results S and index to obove training samples and test data.
    load('PaviaU_traintestsets_10.mat','mtrainingIndex','mtestingIndex','stdIndex');
    trIndex = [];
    tstIndex = [];
    for i =1:length(mtestingIndex)
        curIndex = mtestingIndex(i);
        tstIndex = [tstIndex, find(index==stdIndex(curIndex))];
    end
    for j=1:length(mtrainingIndex)
        curIndex = mtrainingIndex(j);
        trIndex = [trIndex, find(index==stdIndex(curIndex))];    
    end
    mtrainingData = S(trIndex,:);
    mtrainingLabels = vLabels(trIndex,:); 
    mtestingData = S(tstIndex,:);
    mtestingLabels = vLabels(tstIndex,:);  
    
    mtrainingLabels = double(mtrainingLabels);
    mtestingLabels = double(mtestingLabels);  
    %% classify
    [ predicted_label, rr, prob_estimates ] = SVMClassify( mtrainingLabels, mtrainingData, mtestingLabels,mtestingData ); 

    index = double(index);
    resultMap = vgroundTruth;
    resultMap(index(tstIndex)) = predicted_label;
    figure, imagesc(reshape(resultMap,[m,n]));

    [overall,kappa,average,classindividual] = calcError(mtestingLabels'-1,predicted_label'-1,[1:numofClass]);
    resultsFile = sprintf('results_%s_%0.2f.mat',filename,sampleRate);
    save(resultsFile, 'overall','kappa','average','classindividual','predicted_label','mtestingLabels');       
    
end


