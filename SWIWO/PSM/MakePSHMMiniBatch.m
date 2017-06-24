function [ batch, actualSz ] = MakePSHMMiniBatch( dataset, caseData, negUserPr, nNegUsers, negItemPr, nNegItems, batchIdx, batchSz )
startIdx = (batchIdx - 1) * batchSz + 1;
endIdx = min(batchIdx * batchSz, length(caseData));
batchData = caseData(startIdx:endIdx, :);
actualSz = endIdx - startIdx + 1;
dataset = dataset(batchData(:, 1));

items = cell(1, actualSz);
choices = zeros(1, actualSz);

parfor i=1:actualSz
    choice = batchData(i,3);
    session = dataset{i}.Session;
    choices(i) = session(choice);
    session(choice) = [];
    items{i} = session;
end
    
batch.Users = batchData(:, 2);
batch.Items = items;
batch.Choices = choices;

netItems = gpuArray.colon(1,length(negItemPr));
cumNegItem = cumsum(negItemPr).';
netUsers = gpuArray.colon(1,length(negUserPr));
cumNegUser = cumsum(negUserPr).';
batch.ItemNSamp = reshape(netItems( sum( gpuArray.rand(nNegItems * actualSz, 1) > cumNegItem, 2) + 1 ), nNegItems, actualSz);
batch.UserNSamp = reshape(netUsers( sum( gpuArray.rand(nNegUsers * actualSz, 1) > cumNegUser, 2) + 1 ), nNegUsers, actualSz);

end

