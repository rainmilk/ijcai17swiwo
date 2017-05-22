function [ grad, gradhist] = AdagradUpdate( grad, gradhist, stepSize)

if nargin < 3, stepSize = 0.01; end

gradhist = gradhist + grad.^2;
grad = stepSize.*grad./(sqrt(gradhist) + eps);
    
end