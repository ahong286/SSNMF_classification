
addpath(fullfile(pwd, 'Deng Cai_code'));
addpath(fullfile(pwd, 'NMF_denoising'));
addpath(fullfile(pwd, 'Datasets/Pavia'));
addpath(fullfile(pwd, 'libsvm-3.20/matlab'));
datasetname = 'PaviaU';
method = 'SC_CS';

isTrainDictionary = 0;
isSC_CS_Classify = 1;
DataCube = importdata('PaviaU.mat');
groundTruth = importdata('PaviaU_gt.mat');
[m, n, b] = size(DataCube);
DataCube = double(DataCube);
min_val=min(min(min(DataCube)));
max_val=max(max(max(DataCube)));
DataCube=(DataCube-min_val)/(max_val-min_val);

vDataCube =  reshape(DataCube, [m*n,b]);
% vDataCube = NormalizeFea(vDataCube);
vgroundTruth = reshape(groundTruth, [numel(groundTruth),1]);

% patch size
patch_size = [5, 5];
nBasis = 30;  %round(2.0*prod(patch_size));

% SC-CS's parameters;
beta = 0.1;
nIters = 100;
% class number
numofClass = max(groundTruth(:));   % It is suitable for Indian_pines data;

%% Extract the traing data's patches and train codebook; 
if(isTrainDictionary==1)
    index = [];
    for c = 1: numofClass
        cc  = double(c);
        class = find(vgroundTruth == c);
        if isempty(class)
            continue;
        end     
        index = [index;class];
    end
    perm = randperm(numel(index));
    index = index(perm);
    vDicData = vDataCube(index,:);
    vLabels = vgroundTruth(index,:);
    [trDicpatches] = ExtractPatchesByIndex(DataCube,index,patch_size );
    
    [B, S, stat] = sc_cs_hyperspectral(trDicpatches, nBasis, beta, nIters); 
    savefile = sprintf('%s_%s_patches%d_nBasis%d_beta%0.2f_whole.mat',datasetname,method, patch_size(1),nBasis,beta);
    save(savefile,'B','S','vLabels','index');
    clear vtrData trDicpatches W_dic;
end

%% classify based on the sparse codes and denoising results.
if (isSC_CS_Classify == 1)
    savefile = sprintf('%s_%s_patches%d_nBasis%d_beta%0.2f_whole.mat',datasetname,method, patch_size(1),nBasis,beta);
    load(savefile,'S','vLabels','index');
    S = S';
    trainingSamples = cell(numofClass,1);
    testingSamples = cell(numofClass,1);
    trainingIndex = cell(numofClass,1);
    testingIndex = cell(numofClass,1);
    trainingLabels = cell(numofClass,1);
    testingLabels = cell(numofClass,1);
    numofTest = zeros(numofClass,1);

%     sampleRateList = [0.05, 0.1, 0.25];
    sampleRateList = [0.09];
    timeofRepeatition = 1;

    for repeat = 1:timeofRepeatition
    for i = 1 : length(sampleRateList)
        sampleRate = sampleRateList(i);
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
        
        mtrainingLabels = double(mtrainingLabels);
        mtestingLabels = double(mtestingLabels);
        %% classify
        [ predicted_label, rr, prob_estimates ] = SVMClassify( mtrainingLabels, mtrainingData, mtestingLabels,mtestingData ); 
        accuracy(repeat, i) = rr(1);
        
%          [ predicted_label, rr, prob_estimates ] = SVMClassify( mtrainingLabels, mtrainingData_D,mtestingLabels,mtestingData_D ); 
%         accuracy(tempRaw, 2) = rr(1);        
% 
%         [ predicted_label, rr, prob_estimates ] = SVMClassify( mtrainingLabels, mtrainingData_Mix,mtestingLabels,mtestingData_Mix );
%          accuracy(tempRaw, 3) = rr(1);
      
        index = double(index);
        resultMap = vgroundTruth;
        resultMap(index(mtestingIndex)) = predicted_label;
        figure, imagesc(reshape(resultMap,[m,n]));
        % accurancy in each class
        resultC = predicted_label == mtestingLabels;
        for c = 1:numofClass
            accuracyC(c,i,repeat) = sum(resultC(find(mtestingLabels == c)))/numofTest(c);
        end  
    end
    end
    mu = mean(accuracy,2);
    sigma = std(accuracy, 0, 2);
    [overall,kappa,average,classindividual] = calcError(mtestingLabels'-1,predicted_label'-1,[1:numofClass]);
    resultsFile = sprintf('results_%s_%s_patches%d_nBasis%d_beta%0.2f_samplerate%0.2f.mat',datasetname,method, patch_size(1),nBasis,beta,sampleRate);
    save(resultsFile, 'overall','kappa','average','classindividual','predicted_label','mtestingLabels');   
end


