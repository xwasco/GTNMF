function [ A ] = calcSymMatrix( X, graph_type, similarity_type,graph_objfun,kk,nn)
D = dist2(X,X);
n=size(X,1);

if nargin<5 || isempty(kk)
    kk = floor(log2(n)) + 1;
end

if nargin<6 || isempty(nn)
    nn = 7;
end


if strcmp(graph_type, 'full') & strcmp(similarity_type, 'gaussian')
    A = scale_dist3(D, nn);
elseif strcmp(graph_type, 'full') & strcmp(similarity_type, 'inner_product')
    A = X * X';
elseif strcmp(graph_type, 'sparse') & strcmp(similarity_type, 'gaussian')
    if numel(kk)==1
        A = scale_dist3_knn(D, nn, kk, true);
    else
        A = sv_scale_dist3_knn(D, nn, kk, true);
    end
else % graph_type == 'sparse' & similarity_type == 'inner_product'
    Xnorm = X';
    d = 1./sqrt(sum(Xnorm.^2));
    Xnorm = bsxfun(@times, Xnorm, d);
    if numel(kk)==1
        A = inner_product_knn(D, Xnorm, kk, true);
    else
        A = sv_inner_product_knn(D, Xnorm, kk, true);
    end
    clear Xnorm, d;
end

clear D;

if strcmp(graph_objfun, 'ncut')
    dd = 1 ./ sum(A);
    dd = sqrt(dd);
    A = bsxfun(@times, A, dd);
    A = A';
    A = bsxfun(@times, A, dd);
    clear dd;
end
A = (A + A') / 2;
end

