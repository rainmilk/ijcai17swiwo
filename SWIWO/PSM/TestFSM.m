nFeat = 15; nItem = 20;
ExpDecayRt = 0.7;
dataset = cell(1, 100);
parfor i=1:length(dataset)
    dataset{i}.Session = [mod(i, nItem) + 1, mod(i+1, nItem) + 1, mod(i+2, nItem) + 1];
end
FeatureDict = ones(nFeat, nItem);
caseData = MakeCFNetDataCases(dataset);


Epoch = 30;
WtDecay = 0.001;
WtDecayFeat = 0.1;
batchSz = 10;
nNegItems = 5;
nUhid = 10; nIHid = 50;

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

Wf = 0.01 * gpuArray.randn(nUhid, nFeat);
Wi = 0.01 * gpuArray.randn(nIHid, nItem);
Wc = 0.01 * gpuArray.randn(nItem, nUhid + nIHid);
Gf = InitAdamParam; 
Gi = InitAdamParamEx(size(Wi)); 
Gc = InitAdamParamEx(size(Wc));

regs = gpuArray([WtDecayFeat .* ones(1, nUhid), WtDecay .* ones(1, nIHid)]);

nbatch = ceil(nCases / batchSz);
for ep = 1:Epoch
    fprintf('\nEpoch %d starting...', ep);
    
    caseData = caseData(randperm(nCases), :);
    for i=1:nbatch
        % output choice
        [batch, actualSz] = MakeFSMMiniBatch(dataset, caseData, FeatureDict, negItemPr, nNegItems, i, batchSz);
        [ gradWf, gradWi, uniItem, gradWc, uniCh, gradWcNeg, uniChNeg ] = ...
            FSM(batch.Features, batch.Choices, batch.ChoicePos, batch.Items, [], batch.ItemNSamp, Wf, Wi, Wc, ExpDecayRt);
        
        [ grad, Gi ] = AdamUpdateByDim2( gradWi - WtDecay .* Wi(:, uniItem), Gi, uniItem);
        Wi(:, uniItem) = Wi(:, uniItem) + grad;
        
        [ grad, Gf ] =  AdamUpdate( gradWf - WtDecay .* Wf, Gf);
        Wf = Wf + grad;
        
        [ grad, Gc ] = AdamUpdateByDim1( gradWc - regs .* Wc(uniCh, :), Gc, uniCh);
        Wc(uniCh, :) = Wc(uniCh, :) + grad;
        
        [ grad, Gc ] = AdamUpdateByDim1( gradWcNeg - regs .* Wc(uniChNeg, :), Gc, uniChNeg);
        Wc(uniChNeg, :) = Wc(uniChNeg, :) + grad;
    end
end


items = {[8,10], [1,3], [3,5], [11,13]}; pos = [2,2,2,2];
feathid = gpuArray.zeros(size(Wf, 1), length(items));
itemhid = gpuArray.zeros(size(Wi,1), length(items));
for i=1:length(items)
    wts = ExpDecay(pos(i), length(items{i}), ExpDecayRt);
    hidF = sum(FeatureDict(:, items{i}) .* wts, 2);
    feathid(:, i) = logisticfun(Wf * hidF);
    hidI = logisticfun(Wi(:, items{i}));
    itemhid(:, i) = sum(hidI .* wts, 2);
end
mid = [feathid; itemhid];
scores = Wc * mid;
[~, rankFeaItems] = sort(scores, 'descend');