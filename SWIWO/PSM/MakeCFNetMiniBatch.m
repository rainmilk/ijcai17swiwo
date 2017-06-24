function [ batch, actualSz ] = MakeCFNetMiniBatch( dataset, caseData, featDict, negItemPr, nNegSamps, batchIdx, batchSz )
startIdx = (batchIdx - 1) * batchSz + 1;
endIdx = min(batchIdx * batchSz, length(caseData));
batchData = caseData(startIdx:endIdx, :);
actualSz = endIdx - startIdx + 1;
dataset = dataset(batchData(:, 1));

itemNSamp = zeros(nNegSamps, actualSz);
items = cell(1, actualSz);
features = zeros(size(featDict, 1), actualSz);

for i=1:actualSz
    itemNSamp(:, i) = randsample(1:length(negItemPr), nNegSamps, true, negItemPr);
    choiceIdx = batchData(i,3);
    session = dataset{i}.Session;
    session(choiceIdx) = [];
    items{i} = session;
    features(:, i) = any(featDict(:, session), 2);
end

batch.Choices = batchData(:, 2);
batch.Items = items;
batch.Features = features;
batch.ItemNSamp = itemNSamp;

end

