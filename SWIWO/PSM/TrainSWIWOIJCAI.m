Epoch = 50;
WtDecay = 0.001;
ExpDecayRt = 0.7;
batchSz = 200;
nNegItems = 50;
nUhid = 10; nIHid = 50;

aplha = 1;
negItemPr = aplha .* ones(nItem, 1);

nCases = size(trainData, 1);
for i=1:nCases
    userIdx = trainData(i,2);
    itemIdx = trainData(i,4);
    negItemPr(itemIdx) = negItemPr(itemIdx) + 1;
end

negItemPr = log2(negItemPr);
negItemPr = negItemPr / sum(negItemPr);

Wu = 0.01 * gpuArray.randn(nUhid, nUser);
Wi = 0.01 * gpuArray.randn(nIHid, nItem);
Wc = 0.01 * gpuArray.randn(nItem, nUhid + nIHid);
Gu = InitAdamParamEx(size(Wu)); 
Gi = InitAdamParamEx(size(Wi)); 
Gc = InitAdamParamEx(size(Wc));

nbatch = ceil(nCases / batchSz);
start = tic;
for ep = 1:Epoch
    fprintf('\nEpoch %d starting...', ep);
    trainData = trainData(randperm(nCases), :);
    for i=1:nbatch
        % output choice
        [batch, actualSz] = MakePSMMiniBatch(trainingSet, trainData, negItemPr, nNegItems, i, batchSz);
        [ gradWu, uniUser, gradWi, uniItem, gradWc, uniCh, gradWcNeg, uniChNeg ] = ...
            SWIWO(batch.Users, batch.Choices, batch.ChoicePos, batch.Items, [], batch.ItemNSamp, Wu, Wi, Wc, ExpDecayRt);
        [ grad, Gu ] = AdamUpdateByDim2( gradWu - 0.1 .* Wu(:, uniUser), Gu, uniUser);
        Wu(:, uniUser) = Wu(:, uniUser) + grad;
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

rank = zeros(1, nTest);
nTest = length(testllo);
rankllo = zeros(nTest, 1);
itemllo = zeros(nTest, 10);
for i=1:nTest
    hidU = logisticfun(Wu(:, testllo{i}.User));
    hidI = logisticfun(Wi(:, testllo{i}.Session));
    wts = ExpDecay(testllo{i}.Position, length(testllo{i}.Session), ExpDecayRt);
    hidI = sum(hidI .* wts, 2);
    scores = Wc * [hidU; hidI];
    scores = scores';
    [~, Idx] = sort(scores, 'descend');
    Idx = gather(Idx);
    rank(Idx) = 1:nItem;
    rankllo(i) = rank(testllo{i}.TestCase);
    itemllo(i,:) = Idx(1:10);
end
recallAt10_LLO = mean(rankllo<=10);
recallAt20_LLO = mean(rankllo<=20);
recallAt50_LLO = mean(rankllo<=50);
mrr_LLO = mean(1./rankllo);
auc_LLO = mean( (nItem - rankllo) ./ nItem);
cov_LLO = length(unique(itemllo(:))) ./ nItem;
div_LLO = Diversity(itemllo);

nTest = length(testLast);
rankLast = zeros(nTest, 1);
itemLast = zeros(nTest, 10);
for i=1:nTest
    hidU = logisticfun(Wu(:, testLast{i}.User));
    hidI = logisticfun(Wi(:, testLast{i}.Session));
    wts = ExpDecay(testLast{i}.Position, length(testLast{i}.Session), ExpDecayRt);
    hidI = sum(hidI .* wts, 2);
    scores = Wc * [hidU; hidI];
    scores = scores';
    [~, Idx] = sort(scores, 'descend');
    Idx = gather(Idx);
    rank(Idx) = 1:nItem;
    rankLast(i) = rank(testLast{i}.TestCase);
    itemLast(i,:) = Idx(1:10);
end
recallAt10_Last = mean(rankLast<=10);
recallAt20_Last = mean(rankLast<=20);
recallAt50_Last = mean(rankLast<=50);
mrr_Last = mean(1./rankLast);
auc_Last = mean( (nItem - rankLast) ./ nItem);
cov_Last = length(unique(itemLast(:))) ./ nItem;
div_Last = Diversity(itemLast);
