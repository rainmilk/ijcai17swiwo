function [ gradWu, uniUser, gradWi, uniItem, gradWc, uniCh, gradWcNeg, uniChNeg ] = SWIWO( users, choices, choiceidx, sessions, negProbs, negIdx, Wu, Wi, Wc, decayRt)
% outidxs: index outputs of a mini-batch
% negProb: negative samples probability
% size(negIdx) = [K, miniBatchSize]
% size(Wa) = [nFeatures, miniBatchSize]
% size(Wb) = [nOutputUnits, nFeatures]
% size(Biasb) = [miniBatchSize, 1]

NegWc = reshape(Wc(negIdx, :), size(negIdx,1), size(Wc,2), size(negIdx,2));
Wc = Wc(choices, :);
% NegBiasb = Biasb(negIdx);
% Biasb = Biasb(outIdxs);

%% Forward Pass
uniformCB = isempty(choiceidx);
batchSz = length(sessions);
uHid = logisticfun(Wu(:,users));
iHid = gpuArray.zeros(size(Wi, 1), batchSz);
allItem = cell2mat(sessions);
idxSession = zeros(length(allItem));
startIdx = 1;
for i=1:batchSz
    sessLen = length(sessions{i});
    if uniformCB
        wts = 1/sessLen;       
    else
        wts = ExpDecay( choiceidx(i), sessLen, decayRt);
    end
    hid = logisticfun(Wi(:,sessions{i})) .* wts;
    iHid(:,i) = sum(hid, 2); 
    endIdx = startIdx + sessLen - 1;
    idxSession(startIdx:endIdx, 1) = i;
    idxSession(startIdx:endIdx, 2) = wts;
    startIdx = endIdx + 1;
end
hid = [uHid; iHid];
net = sum(Wc' .* hid);
netNeg = sum(NegWc .* permute(hid, [3,1,2]), 2);

%% backward pass
if isempty(negProbs)
    % Negative sampling for approximating softmax
    WP_out = 1 - logisticfun(net);
    WNeg_out = logisticfun(netNeg);
else
    % NCE for approximating softmax
    K = size(negIdx, 1);
    logPnP = log(K*negProbs(choices))';
    logPnN = log(K*negProbs(negIdx));
    WP_out = 1 - logisticfun(net - logPnP);
    WNeg_out = logisticfun(netNeg - permute(logPnN, [1,3,2]));
end

gradWc = hid .* WP_out;
gradWcNeg = -permute(hid, [1,3,2]) .* permute(WNeg_out, [2,1,3]);

d_hid = hid .* (1 - hid);
gradWui = WP_out .* Wc' - permute(pagefun(@mtimes, permute(NegWc, [2,1,3]), WNeg_out), [1,3,2]);
gradWui = gradWui .* d_hid;

gradWu = gradWui(1:size(Wu, 1), :);
gradWi = gradWui(size(Wu, 1)+1:end, :);

[uniUser, ia] = unique(users);
gradWu = gradWu(:, ia);

[uniItem, ia] = unique(allItem);
gradWi = gradWi(:, idxSession(ia,1)) .* idxSession(ia,2)';

[uniCh, ia] = unique(choices);
gradWc = gradWc(:, ia)';

[uniChNeg, ia] = unique(negIdx(:));
gradWcNeg = gradWcNeg(:, ia)';

end


