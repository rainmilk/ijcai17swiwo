function [ grad, gradhist] = RMSPropUpdate( grad, gradhist, stepSize, gamma)

if nargin < 3, stepSize = 0.01; end
if nargin < 4, gamma = 0.9; end

gradhist = gamma.*gradhist + (1-gamma).*(grad.^2);
grad = stepSize.*grad ./ (sqrt(gradhist) + eps);
    
end