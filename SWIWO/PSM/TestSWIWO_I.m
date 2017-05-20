nItem = 20;
dataset = cell(1, 200);
userIdx = ceil(nUser*rand(length(dataset),1));
for i=1:length(dataset)
    dataset{i}.User = userIdx(i);
    session = [mod(i, nItem) + 1, mod(i+1, nItem) + 1, mod(i+2, nItem) + 1];
    dataset{i}.Session = session;
end
caseData = MakePSHMDataCases(dataset);

Epoch = 20;
WtDecay = 0.01;
ExpDecayRt = 0.8;
batchSz = 10;
nNegItems = 5;
nIHid = 50;

aplha = 1;
negItemPr = aplha .* ones(nItem, 1);

nCases = size(caseData, 1);
for i=1:nCases
    userIdx = caseData(i,2);
    itemIdx = caseData(i,4);
    negItemPr(itemIdx) = negItemPr(itemIdx) + 1;
end

negItemPr = log2(negItemPr);
negItemPr = negItemPr / sum(negItemPr);

Wi = 0.01 * gpuArray.randn(nIHid, nItem);
Wc = 0.01 * gpuArray.randn(nItem, nIHid); 
Gi = InitAdamParamEx(size(Wi)); 
Gc = InitAdamParamEx(size(Wc));

nbatch = ceil(nCases / batchSz);
start = tic;
for ep = 1:Epoch
    fprintf('\nEpoch %d starting...', ep);
    caseData = caseData(randperm(nCases), :);
    for i=1:nbatch
        % output choice
        [batch, actualSz] = MakePSMMiniBatch(dataset, caseData, negItemPr, nNegItems, i, batchSz);
        [ gradWi, uniItem, gradWc, uniCh, gradWcNeg, uniChNeg ] = ...
            SWIWO_I(batch.Choices, [], batch.Items, [], batch.ItemNSamp, Wi, Wc, ExpDecayRt);
        [ grad, Gi ] = AdamUpdateByDim2( gradWi - WtDecay .* Wi(:, uniItem), Gi, uniItem);
        Wi(:, uniItem) = Wi(:, uniItem) + grad;

        [ grad, Gc ] = AdamUpdateByDim1( gradWcNeg - WtDecay .* Wc(uniChNeg, :), Gc, uniChNeg);
        Wc(uniChNeg, :) = Wc(uniChNeg, :) + grad;
        [ grad, Gc ] = AdamUpdateByDim1( gradWc - WtDecay .* Wc(uniCh, :), Gc, uniCh);
        Wc(uniCh, :) = Wc(uniCh, :) + grad;
    end
    dtm = toc(start);
    fprintf('\nEpoch %d finished, est. in %g secs', ep, (Epoch-ep)*dtm/ep);
end

items = {[7,8], [1,2], [3,5], [11,13]}; pos = [3,3,2,2];
itemhid = gpuArray.zeros(size(Wi,1), length(items));
for i=1:length(items)
    len = length(items{i});
    hid = logisticfun(Wi(:, items{i}));
    % wts = ExpDecay(pos(i), len); 
    wts = 1/len;
    itemhid(:, i) = sum(hid .* wts, 2);
end
scores = Wc * itemhid;
[~, rankItems] = sort(scores, 'descend');