Epoch = 50;
WtDecay = 0.001;
ExpDecayRt = 0.7;
batchSz = 200;
nNegItems = 50;
nIHid = 50;

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

Wi = 0.01 * gpuArray.randn(nIHid, nItem);
Wc = 0.01 * gpuArray.randn(nItem, nIHid); 
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
        [ gradWi, uniItem, gradWc, uniCh, gradWcNeg, uniChNeg ] = ...
            SWIWO_I(batch.Choices, batch.ChoicePos, batch.Items, [], batch.ItemNSamp, Wi, Wc, ExpDecayRt);
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

rank = zeros(1, nItem);
nTest = length(testllo);
rank_I_llo = zeros(nTest, 1);
item_I_llo = zeros(nTest, 10);
for i=1:nTest
    hidI = logisticfun(Wi(:, testllo{i}.Session));
    wts = ExpDecay(testllo{i}.Position, length(testllo{i}.Session), ExpDecayRt);
    hidI = sum(hidI .* wts, 2);
    scores = Wc * hidI;
    [~, Idx] = sort(scores, 'descend');
    Idx = gather(Idx);
    rank(Idx) = 1:nItem;
    rank_I_llo(i) = rank(testllo{i}.TestCase);
    item_I_llo(i,:) = Idx(1:10);
end
recallAt10_I_LLO = mean(rank_I_llo<=10);
recallAt20_I_LLO = mean(rank_I_llo<=20);
recallAt50_I_LLO = mean(rank_I_llo<=50);
mrr_I_LLO = mean(1./rank_I_llo);
auc_I_LLO = mean( (nItem - rank_I_llo) ./ nItem);
cov_I_LLO = length(unique(item_I_llo(:))) ./ nItem;

nTest = length(testLast);
rank_I_Last = zeros(nTest, 1);
item_I_Last = zeros(nTest, 10);
for i=1:nTest
    hidI = logisticfun(Wi(:, testLast{i}.Session));
    wts = ExpDecay(testLast{i}.Position, length(testLast{i}.Session), ExpDecayRt);
    hidI = sum(hidI .* wts, 2);
    scores = Wc * hidI;
    scores = scores';
    [~, Idx] = sort(scores, 'descend');
    Idx = gather(Idx);
    rank(Idx) = 1:nItem;
    rank_I_Last(i) = rank(testLast{i}.TestCase);
    item_I_Last(i,:) = Idx(1:10);
end
recallAt10_I_Last = mean(rank_I_Last<=10);
recallAt20_I_Last = mean(rank_I_Last<=20);
recallAt50_I_Last = mean(rank_I_Last<=50);
mrr_I_Last = mean(1./rank_I_Last);
auc_I_Last = mean( (nItem - rank_I_Last) ./ nItem);
cov_I_Last = length(unique(item_I_Last(:))) ./ nItem;

div_I_LLO = Diversity(item_I_llo);
div_I_Last = Diversity(item_I_Last);