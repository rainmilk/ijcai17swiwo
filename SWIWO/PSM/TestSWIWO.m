nUser = 5; nItem = 20;
dataset = cell(1, 200);
userIdx = ceil(nUser*rand(length(dataset),1));
for i=1:length(dataset)
    dataset{i}.User = userIdx(i);
    session = [mod(i, nItem) + 1, mod(i+1, nItem) + 1, mod(i+2, nItem) + 1];
    dataset{i}.Session = session;
end
caseData = MakeSWIWODataCases(dataset);

Epoch = 20;
WtDecay = 0.01;
batchSz = 10;
nNegItems = 5;
nUhid = 50; nIHid = 50;

aplha = 1;
negItemPr = aplha .* ones(nItem, 1);

nCases = size(caseData, 1);
for i=1:nCases
    userIdx = caseData(i,2);
    itemIdx = caseData(i,4);
    negItemPr(itemIdx) = negItemPr(itemIdx) + 1;
end

negItemPr = negItemPr / sum(negItemPr);
negItemPr = negItemPr.^0.5;
negItemPr = negItemPr / sum(negItemPr);

Wu = 0.01 * gpuArray.randn(nUhid, nUser);
Wi = 0.01 * gpuArray.randn(nIHid, nItem);
Wc = 0.01 * gpuArray.randn(nItem, nUhid + nIHid);
Gu = InitAdagradParam(size(Wu)); 
Gi = InitAdagradParam(size(Wi)); 
Gc = InitAdagradParam(size(Wc));

nbatch = ceil(nCases / batchSz);
for ep = 1:Epoch
    fprintf('\nEpoch %d starting...', ep);
    
    caseData = caseData(randperm(nCases), :);
    for i=1:nbatch
        % output choice
        [batch, actualSz] = MakePSMMiniBatch(dataset, caseData, negItemPr, nNegItems, i, batchSz);
        [ gradWu, uniUser, gradWi, uniItem, gradWc, uniCh, gradWcNeg, uniChNeg ] = ...
            SWIWO(batch.Users, batch.Choices, batch.ChoicePos, batch.Items, negItemPr, batch.ItemNSamp, Wu, Wi, Wc);
        [ grad, Gu(:, uniUser) ] = RMSPropUpdate( gradWu - WtDecay .* Wu(:, uniUser), Gu(:, uniUser));
        Wu(:, uniUser) = Wu(:, uniUser) + grad;
        [ grad, Gi(:, uniItem) ] = RMSPropUpdate( gradWi - WtDecay .* Wi(:, uniItem), Gi(:, uniItem));
        Wi(:, uniItem) = Wi(:, uniItem) + grad;
        [ grad, Gc(uniCh, :) ] = RMSPropUpdate( gradWc - WtDecay .* Wc(uniCh, :), Gc(uniCh, :));
        Wc(uniCh, :) = Wc(uniCh, :) + grad;
        [ grad, Gc(uniChNeg, :) ] = RMSPropUpdate( gradWcNeg - WtDecay .* Wc(uniChNeg, :), Gc(uniChNeg, :));
        Wc(uniChNeg, :) = Wc(uniChNeg, :) + grad;
    end
end

uidx = [3,1,1,2]; items = {[8], [1,2], [3,5], [11,13]};
userhid = logisticfun(Wu(:, uidx));
itemhid = gpuArray.zeros(size(Wi,1), length(items));
for i=1:length(items)
    len = length(items{i});
    hid = logisticfun(Wi(:, items{i}));
    wts = ExpDecay(len + 1, len); 
    itemhid(:, i) = sum(hid .* wts, 2);
end
mid = [userhid; itemhid];
scores = Wc * mid;
[~, rankItems] = sort(scores, 'descend');