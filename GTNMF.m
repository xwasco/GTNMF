function [HC,SC] = GTNMF(S,H,GT,knn,maxIter,maxDiff)
%% GTNMF Game theoretic NMF clustering refiner
%
% Input:
% S         the similarity matrix of the dataset (nxn) with zero on the main diagonal
%
% H         the soft-assignment clustering of NMF (nxk)
%
% GT        (nx1) groundtruth hard assignment, if provided the method plots
%           the performances per step. The GT is used only to evaluate the
%           performances and is not part of the method.
%
% knn       if knn=0 imply no sparsification, if knn=-1 use the floor(log2(N))+1
%           Nearest neighbours if knn>0 use that amount of nearest
%           neighbours required (default -1). We strongly reccomend to sparsify the matrix 
%           if the matrix is not already sparsifed.
%
% maxIter   the maximum number of iterations (default 200)
%
% maxDiff   the maximum difference between a step and the next (default 1e-5)
%
% Output:
% HC        the hard cluster assignment (max of the soft clustering)
%
% SC        the soft cluster assignment
%
% Please if you use this code cite this paper:
% Tripodi Rocco *, Vascon Sebastiano * and Pelillo Marcello - Context Aware Nonnegative Matrix Factorization Clustering
% International Conference on Pattern Recognition (ICPR) 2016
%
%%%%%%%%%%%%%%%%%%%%%
    
plotGraph=0;    
if nargin<3
    GT=[];
else
    if ~isempty(GT)
        plotGraph=1;
        tempS=S+1;
        tempS=tempS-1;
    end
end

if nargin<4
    knn=-1;
end

if nargin<5
    maxIter=200;
end

if nargin<6
    maxDiff=1e-5;
end

if knn==-1
    knn=floor(log2(size(S,1)))+1;
end

if knn>0
    %sparsify the similarity matrix using a NN rule
    [~,IDX]=sort(S,2,'descend');
    IDX=IDX(:,[1:knn]);
    for i=1:size(S,1)-1
        for j=i+1:size(S,1)
            if ~ismember(i,IDX(j,:)) && ~ismember(j,IDX(i,:))
                S(i,j)=0;
                S(j,i)=0;
            end
        end
    end
end

niter = 0;
accs=[];

SC=H+min(min(H));
pSum = sum(SC,2)+eps;
SC = bsxfun(@rdivide, SC, pSum);
    
if plotGraph
    hh=figure;
    [~,clasI2] = max(SC,[],2);
    [ACC,CM]=classificationAccuracy(GT,clasI2);
    [MI]=mi(CM);
    accs=[accs ; [niter, ACC, MI]];
    diffs=[];
end

while true,
    q = S*SC;
    dummy = SC.*q;
    dummySum = sum(dummy,2)+eps;
    pnew = bsxfun(@rdivide, dummy, dummySum);
    
    diff = norm(SC(:)-pnew(:));
    
    SC = pnew;
    niter = niter+1 ;
    
    if plotGraph
        [~,clasI2] = max(SC,[],2);
        [ACC,CM]=classificationAccuracy(GT,clasI2);
        [MI]=mi(CM);
        accs=[accs ; [niter, ACC, MI]];
        diffs=[diffs ; diff];
        
        figure(hh);
        
        subplot(2,4,1);
        imagesc(H);
        title('SoftAssignment');    
        
        subplot(2,4,2);
        imagesc(tempS);
        title('Similarity Matrix');   
        
        subplot(2,4,3);
        imagesc(GT);
        title('Ground Truth');
        
        subplot(2,4,5);
        imagesc(SC);
        title('SoftAssignment(refined)');
        
        subplot(2,4,6);
        imagesc(S);
        title('Similarity Matrix (sparse)');   
        
        subplot(2,4,7);
        plot(diffs);
        title('Step Error');
        
        subplot(2,4,8);
        ax =plotyy(accs(:,1),accs(:,2),accs(:,1),accs(:,3));
        xlabel('Iterations');
        ylabel(ax(1), 'Accuracy');
        ylabel(ax(2), 'Normalized Mutual Information');
        title('Performances');          
        drawnow ;
    end    
    
    if niter==maxIter || diff<maxDiff
        break;
    end
end
[~,HC] = max(SC,[],2);

fprintf(['***************************\nIf you use this code please cite this paper:\nR. Tripodi, S.Vascon and M.Pelillo\nContext Aware Nonnegative Matrix Factorization Clustering\nInternational Conference of Pattern Regonition (ICPR) 2016\n***************************\n']);


end

