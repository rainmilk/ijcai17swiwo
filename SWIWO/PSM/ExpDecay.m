function [ wts ] = ExpDecay( ci, ctxLen, lambda )
if nargin < 3, lambda = .7; end

wts = zeros(1, ctxLen);
if ci > 1
    wts(1:ci-1) = (ci-2):-1:0;
end

if ci <= ctxLen
    wts(ci:end) = 0:(ctxLen-ci);
end

wts = exp(-lambda * wts);
wts = wts ./ sum(wts);

end

