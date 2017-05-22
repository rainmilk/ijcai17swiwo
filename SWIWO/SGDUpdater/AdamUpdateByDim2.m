function [ grad, params ] = AdamUpdateByDim2( grad, params, idx, stepSize, beta1, beta2)

if nargin < 4, stepSize = 0.01; end
if nargin < 5, beta1 = 0.9; end
if nargin < 6, beta2 = 0.999; end

% Update biased 1st moment estimate
params.m(:,idx) = beta1.*params.m(:,idx) + (1 - beta1).*grad;
% Update biased 2nd raw moment estimate
params.v(:,idx) = beta2.*params.v(:,idx) + (1 - beta2).*(grad.^2);

params.beta1pow(:,idx) = params.beta1pow(:,idx) * beta1;
params.beta2pow (:,idx)= params.beta2pow(:,idx) * beta2;

% Compute bias-corrected 1st moment estimate
mHat = params.m(:,idx) ./ (1 - params.beta1pow(:,idx));
% Compute bias-corrected 2nd raw moment estimate
vHat = params.v(:,idx) ./ (1 - params.beta2pow(:,idx));

grad = stepSize.*mHat ./ (sqrt(vHat) + eps);
    
end

