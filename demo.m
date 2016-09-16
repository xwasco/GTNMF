clear all
close all

addpath(genpath('utils'));
addpath(genpath('nmf_anls'));
addpath(genpath('symnmf2'));

%% load data 
%coil20 example
load(['coil.mat']);
IMG=double(IMG);

%orl example
%load(['orl.mat']);
%IMG=double(IMG);

IMG=IMG./(repmat(max(IMG,[],1),size(IMG,1),1)+eps);
numCluster=numel(unique(GT));

%% compute similarity matrix
S=calcSymMatrix(IMG,'sparse','gaussian','ncut');

%% perform NMF clustering over S
[W,~]=nmf(IMG,numCluster,'method','anls_bpp','verbose',0);
[~,clustNmf] = max(W,[],2);
[accNmf,confNmf]=classificationAccuracy(GT,clustNmf);
[miNmf]=mi(confNmf);
fprintf(['NMF \t Acc='  num2str(accNmf) ' , NMI=' num2str(miNmf) '\n']);

%GT refinement
[GTG]=GTNMF(full(S),W,GT,0); %S is already sparsified
[accSymNmfGTG,CM_NMFGTG]=classificationAccuracy(GT,GTG);
[miSymNmfGTG]=mi(CM_NMFGTG);
fprintf(['GT-NMF \t Acc='  num2str(accSymNmfGTG) ' , NMI=' num2str(miSymNmfGTG) '\n']);
fprintf('----------\n');

[W] = symnmf_anls(S, numCluster);
[~,clustNmf] = max(W,[],2);
[acc,CM]=classificationAccuracy(GT,clustNmf);
[minf]=mi(CM);
fprintf(['SymNMF \t Acc='  num2str(acc) ' , NMI=' num2str(minf) '\n']);

%GT refinement
[GTG]=GTNMF(full(S),W,GT,0); %S is already sparsified
[accSymNmfGTG,CM_NMFGTG]=classificationAccuracy(GT,GTG);
[miSymNmfGTG]=mi(CM_NMFGTG);
fprintf(['GT-SymNMF \t Acc='  num2str(accSymNmfGTG) ' , NMI=' num2str(miSymNmfGTG) '\n']);