
ccc
gcp; % start a parallel engine

% Find base repository path (where this file is contained)
% basePath = fileparts(which('demo_MMM_pipeline'));

% Add to MATLAB path
% idcs   = strfind(basePath,filesep);
% addpath(genpath(fullfile(basePath(1:idcs(end)-1))));

% Download dataset using link https://doi.org/10.34770/bzkz-j672 see README

% https://doi.org/10.34770/bzkz-j672



%% Data in Brief

% load
load('jointsCON.mat')

% brief data description
for i=1:size(jointsCON,2)
    disp(['group_', num2str(i), ' - ', num2str(size(jointsCON{1,i},1)),...
        ' ', jointsCON{2,i},' ', jointsCON{3,i},' mice', ' cross ',num2str(size(jointsCON{1,i},2)),' days'])
end

%% select which body parts to use in pipeline

xIdx = [1 2 5 6 7 8 12 13 14 15 16];
yIdx = [1 2 5 6 7 8 12 13 14 15 16];

[X,Y] = meshgrid(xIdx,yIdx);
X = X(:); Y = Y(:);
IDX = find(X~=Y);

% do online PCA to find modes
batchSize = 20000;
firstBatch = true;
currentImage = 0;

for g = 1:size(jointsCON,2)
    for i = 1:size(jointsCON{1,g},1)
        fprintf(1,['Processing data for mouse ' num2str(i) ' from group_' num2str(g) '\n']);
        for j = 1:size(jointsCON{1,g},2)
            if ~isempty(jointsCON{1,g}{i,j})
                
                % calculate distances from body part positions
                p1 = jointsCON{1,g}{i,j};
                p1Dist = zeros(size(X,1),size(p1,3));
                for ii = 1:size(p1Dist,1)
                    p1Dist(ii,:) = returnDist(squeeze(p1(X(ii),:,:)),squeeze(p1(Y(ii),:,:)));
                end
                
                p1Dsmooth = zeros(size(p1Dist));
                for ii = 1:size(p1Dist,1)
                    p1Dsmooth(ii,:) = medfilt1(smooth(p1Dist(ii,:),5),5);
                end
                p1Dist = p1Dsmooth(IDX,:);
                
                if firstBatch
                    firstBatch = false;
                    if size(p1Dist,2)<batchSize
                        cBatchSize = size(p1Dist,2);
                        XX = p1Dist';
                    else
                        cBatchSize = batchSize;
                        XX = p1Dist(:,randperm(size(p1Dist,2),cBatchSize))';
                    end
                    
                    currentImage = batchSize;
                    muv = sum(XX);
                    C = cov(XX).*batchSize + (muv'*muv)./cBatchSize;
                else
                    if size(p1Dist,2)<batchSize
                        cBatchSize = size(p1Dist,2);
                        XX = p1Dist';
                    else
                        cBatchSize = batchSize;
                        XX = p1Dist(:,randperm(size(p1Dist,2),cBatchSize))';
                    end
                    
                    tempMu = sum(XX);
                    muv = muv+tempMu;
                    C = C + cov(XX).*cBatchSize + (tempMu'*tempMu)./cBatchSize;
                    currentImage = currentImage + cBatchSize;
                    pause(.1)
                    fprintf(1,[num2str(currentImage),'\n'])
                end
                
            else
            end
            
        end
    end
end
L = currentImage;
muv = muv./L;
C = C./L - muv'*muv;

[vecs,vals] = eig(C);
vals = flipud(diag(vals));
vecs = fliplr(vecs);
plot(1-(vals./sum(vals)))
prepfig;
xlabel('PCA component')
ylabel('var explained')
save('vecsVals_subSelect_demo.mat','C','muv','currentImage','vecs','vals');

%% Project to PCA space, get wavelets and find data for training
% set papameters
dataAll = [];
numProjections = 10;
numModes = 10; pcaModes = numModes;
minF = .25; maxF = 20;

parameters = setRunParameters([]);
parameters.trainingSetSize = 80;
parameters.pcaModes = pcaModes;
parameters.samplingFreq = 80;
parameters.minF = minF;
parameters.maxF = maxF;
numPerDataSet = parameters.trainingSetSize;
numPoints = 5000;

%% check parameters on single session
g=7; i=1; j=3;

% calculate distances from body part positions
p1 = jointsCON{1,g}{i,j};
p1Dist = zeros(size(X,1),size(p1,3));
for ii = 1:size(p1Dist,1)
    p1Dist(ii,:) = returnDist(squeeze(p1(X(ii),:,:)),squeeze(p1(Y(ii),:,:)));
end

p1Dsmooth = zeros(size(p1Dist));
for ii = 1:size(p1Dist,1)
    p1Dsmooth(ii,:) = medfilt1(smooth(p1Dist(ii,:),5),5);
end
p1Dist = p1Dsmooth(IDX,:);

p2Dist = bsxfun(@minus,p1Dist,muv');
projections = p2Dist'*vecs(:,1:numProjections);

figure;
imagesc(projections(4500:5000,:)');
colormap inferno
prepfig; ylabel('PCA projection'); xlabel('Frames');
[data,~] = findWavelets(projections,numModes,parameters);
amps = sum(data,2);

data=log(data);
figure;
histogram(reshape(data,size(data,1)*size(data,2),1),100,'Normalization','probability'); hold on
data(data<-3) = -3;
histogram(reshape(data,size(data,1)*size(data,2),1),100,'Normalization','probability')
prepfig;
figure;
imagesc(data(4500:5000,:)');
colormap magma
prepfig; yticks([]); xlabel('Frames');

yData = tsne(data(1:10:end,:));
[xx,density,G,Z] = findPointDensity(yData, max(max(yData)) / 40, 501);
figure;
imagesc(xx,xx,density)
axis equal tight off xy
colormap parula
hold on;
plot(yData(450:500,2),yData(450:500,1),'k')
prepfig;
title('Behavioral space density')

[signalData,signalAmps] = findTemplatesFromData(...
    data,yData,amps,numPerDataSet,parameters);
figure;
imagesc(signalData')
colormap magma
prepfig; yticks([]); xlabel('Frames');
title('Wavelets for training')

%% Run parameters on full data set
dataAll = [];

for g = 7 %1:size(jointsCON,2)
    for i = 1:size(jointsCON{1,g},1)
        fprintf(1,['Processing data for mouse ' num2str(i) ' from group_' num2str(g) '\n']);
        for j = 1:size(jointsCON{1,g},2)
            if ~isempty(jointsCON{1,g}{i,j})
                p1 = jointsCON{1,g}{i,j};
                p1Dist = zeros(size(X,1),size(p1,3));
                for ii = 1:size(p1Dist,1)
                    p1Dist(ii,:) = returnDist(squeeze(p1(X(ii),:,:)),squeeze(p1(Y(ii),:,:)));
                end
                
                p1Dsmooth = zeros(size(p1Dist));
                for ii = 1:size(p1Dist,1)
                    p1Dsmooth(ii,:) = medfilt1(smooth(p1Dist(ii,:),5),5);
                end
                p1Dist = p1Dsmooth(IDX,:);
                
                p2Dist = bsxfun(@minus,p1Dist,muv');
                projections = p2Dist'*vecs(:,1:numProjections);
                
                [data,~] = findWavelets(projections,numModes,parameters);
                amps = sum(data,2);
                
                N = length(projections(:,1));
                numModes = parameters.pcaModes;
                skipLength = floor(N / numPoints);
                if skipLength == 0
                    skipLength = 1;
                    numPoints = N;
                else
                end
                
                firstFrame = mod(N,numPoints) + 1;
                signalIdx = firstFrame:skipLength:(firstFrame + (numPoints-1)*skipLength);
                
                signalAmps = amps(signalIdx);
                nnData = log(data(signalIdx,:));
                nnData(nnData<-3) = -3;
                
                yData = tsne(real(nnData));
                [signalData,signalAmps] = findTemplatesFromData(...
                    nnData,yData,signalAmps,numPerDataSet,parameters);
                
                dataAll = [dataAll; signalData];
                
            else
            end
        end
    end
end
dataAll = real(dataAll);

%% Embedding/Clustering
tic
ydata = tsne(dataAll);
toc
tic
C100 = kmeans(dataAll,100,'Replicates',10);
toc
tic
C200 = kmeans(dataAll,200,'Replicates',10);
toc

save('trainingSet_demo.mat','dataAll','ydata','C100','C200');
