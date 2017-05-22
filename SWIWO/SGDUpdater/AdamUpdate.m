function [ grad, params ] = AdamUpdate( grad, params, stepSize, beta1, beta2)

if nargin < 3, stepSize = 0.01; end
if nargin < 4, beta1 = 0.9; end
if nargin < 5, beta2 = 0.999; end

% Update biased 1st moment estimate
params.m = beta1.*params.m + (1 - beta1).*grad;
% Update biased 2nd raw moment estimate
params.v = beta2.*params.v + (1 - beta2).*(grad.^2);

params.beta1pow = params.beta1pow * beta1;
params.beta2pow = params.beta2pow * beta2;

% Compute bias-corrected 1st moment estimate
mHat = params.m ./ (1 - params.beta1pow);
% Compute bias-corrected 2nd raw moment estimate
vHat = params.v ./ (1 - params.beta2pow);

grad = stepSize.*mHat ./ (sqrt(vHat) + eps);
    
end

