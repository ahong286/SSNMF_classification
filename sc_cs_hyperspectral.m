function [B S stat] = sc_cs_hyperspectral(X,num_bases, beta, num_iters, Binit, pars)
% Graph regularized sparse coding algorithms
%
%    minimize_B,S   0.5*||X - B*S||^2 + beta*sum(abs(S(:)))
%    subject to   ||B(:,j)||_2 <= l2norm, forall j=1...size(S,1)
% 
% Notation:
% X: data matrix, each column is a sample vector
% W: affinity graph matrix
% num_bases: number of bases
% alpha: Laplician parameter
% beta: sparsity penalty parameter
% num_iters: number of iteration
% Binit: initial B matrix, K-by-1 cell array containing K M-by-P basis
% matrices 
% pars: additional parameters to specify (see the code)
%
% This code is modified from the codes provided by Honglak Lee, Alexis
% Battle, Rajat Raina, and Andrew Y. Ng in the following paper:
% 'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
% Advances in Neural Information Processing Systems (NIPS) 19, 2007
%
% References:
% [1] Miao Zheng, Jiajun Bu, Chun Chen, Can Wang, Lijun Zhang, Guang Qiu, Deng Cai. 
% "Graph Regularized Sparse Coding for Image Representation",
% IEEE Transactions on Image Processing, Vol. 20, No. 5, pp. 1327-1336, 2011. 
%
% Version1.0 -- Nov/2009
% Version2.0 -- Jan/2012
% Written by Miao Zheng <cauthy AT zju.edu.cn>
%

diff = 1e-7;

pars.mFea = size(X{1},1);
pars.nSmp = size(X{1},2);
pars.num_bases = num_bases;
pars.num_iters = num_iters;
pars.beta = beta;
pars.noise_var = 1;
pars.sigma = 1;
pars.VAR_basis = 1;


% Sparsity parameters
if ~isfield(pars,'tol')
    pars.tol = 0.005;
end
K = length(X);
B = cell(K,1);
% initialize basis
for k = 1:K
    if ~exist('Binit','var') || isempty(Binit)
        B{k} = rand(pars.mFea,pars.num_bases)-0.5;
        B{k} = B{k} - repmat(mean(B{k},1), size(B{k},1),1);
        B{k} = B{k}*diag(1./sqrt(sum(B{k}.*B{k})));
    else
        disp('Using Binit...');
        B = Binit;
    end;
end

% initialize t only if it does not exist
t=0;
% statistics variable
stat= [];
stat.fobj_avg = [];
stat.fresidue_avg = [];
stat.fsparsity_avg = [];
stat.var_tot = [];
stat.svar_tot = [];
stat.elapsed_time=0;

% %======================================================
% % Construct the K-NN Graph with spectral information after the Bai Xiao's denoising  
% if isempty(W)
%     W = constructW(X');
% end
% DCol = full(sum(W,2));
% D = spdiags(DCol,0,speye(size(W,1)));
% L = D - W;
%=================================================

% optimization loop
while t < pars.num_iters
    t=t+1;
    start_time= cputime;
    
    stat.fobj_total=0;
    stat.fresidue_total=0;
    stat.fsparsity_total=0;
    stat.var_tot=0;
    stat.svar_tot=0;
       
    % learn coefficients (conjugate gradient)
    if t ==1
        S= learn_coefficients_cs(B, X, pars.beta/pars.sigma*pars.noise_var);
    else
        S= learn_coefficients_cs(B, X, pars.beta/pars.sigma*pars.noise_var, S);
    end
    S(isnan(S))=0;

     % get objective
    [fobj, fresidue, fsparsity] = getObjective(B, S, X, pars.noise_var, pars.beta, pars.sigma);

    stat.fobj_total      = stat.fobj_total + fobj;
    stat.fresidue_total  = stat.fresidue_total + fresidue;
    stat.fsparsity_total = stat.fsparsity_total + fsparsity;
    stat.var_tot         = stat.var_tot + sum(sum(S.^2,1))/size(S,1);

    % update basis
    for k=1:K
        B{k} = learn_basis(X{k}, S, pars.VAR_basis);
    end
    % get statistics
    stat.fobj_avg(t)      = stat.fobj_total / pars.nSmp;
    stat.fresidue_avg(t)  = stat.fresidue_total / pars.nSmp;
    stat.fsparsity_avg(t) = stat.fsparsity_total / pars.nSmp;
    stat.var_avg(t)       = stat.var_tot / pars.nSmp;
    stat.svar_avg(t)      = stat.svar_tot / pars.nSmp;
    stat.elapsed_time(t)  = cputime - start_time;
    
    
    if t>199
        if(stat.fobj_avg(t-1) - stat.fobj_avg(t)<diff)
            return;
        end
    end
    
    
    fprintf(['epoch= %d, fobj= %f, fresidue= %f, fsparsity= %f, took %0.2f ' ...
             'seconds\n'], t, stat.fobj_avg(t), stat.fresidue_avg(t), ...
            stat.fsparsity_avg(t), stat.elapsed_time(t));
    
end



function [fobj, fresidue, fsparsity] = getObjective(A, S, X, noise_var, beta, sigma)
K = length(X);
E = A{1}*S - X{1};
for k=2:K
    E = E + A{k}*S - X{k};
end
lambda=1/noise_var;
fresidue  = 0.5*lambda*sum(sum(E.^2));
fsparsity = beta*sum(sum(abs(S/sigma)));
fobj = fresidue + fsparsity ;
return