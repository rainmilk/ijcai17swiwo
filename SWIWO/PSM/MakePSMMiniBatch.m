function [ batch, actualSz ] = MakePSMMiniBatch( dataset, caseData, negItemPr, nNegItems, batchIdx, batchSz, inputAsNSamps )
if nargin < 7
    inputAsNSamps = true;
end

startIdx = (batchIdx - 1) * batchSz + 1;
endIdx = min(batchIdx * batchSz, length(caseData));
batchData = caseData(startIdx:endIdx, :);
actualSz = endIdx - startIdx + 1;
dataset = dataset(batchData(:, 1));

items = cell(1, actualSz);
choices = zeros(1, actualSz);
negSamp = zeros(nNegItems, actualSz);
cumNegItem = cumsum(negItemPr).';
netItems = (1:length(negItemPr))';

parfor i=1:actualSz
    choice = batchData(i,3);
    session = dataset{i}.Session;
    choices(i) = session(choice);
    session(choice) = [];
    items{i} = session;
    if inputAsNSamps
        nRemain = nNegItems - length(session);
        if nRemain > 0
            negSamp(:,i) = [session(:); netItems( sum( rand(nRemain, 1) > cumNegItem, 2) + 1 )];
        else
            negSamp(:,i) = session(1:nNegItems);
        end
    else
        negSamp(:,i) = netItems( sum( rand(nNegItems, 1) > cumNegItem, 2) + 1 );        
    end
end
    
batch.Users = batchData(:, 2);
batch.Items = items;
batch.Choices = choices;
batch.ChoicePos = batchData(:, 3);
batch.ItemNSamp = negSamp;

end

