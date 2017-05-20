function [ cases ] = MakePSHMDataCases( dataset )
nSession = length(dataset);
caseLen = ones(1, nSession);
for i=1:nSession
   caseLen(i) = length(dataset{i}.Session);
end

cases = zeros(sum(caseLen), 4);

currIdx = 1;
for i=1:nSession
    len = length(dataset{i}.Session);
    sIdx = currIdx : currIdx + len - 1;
    cases(sIdx, 1) = i;
    cases(sIdx, 2) = dataset{i}.User;
    cases(sIdx, 3) = 1:len;
    cases(sIdx, 4) = dataset{i}.Session;
    currIdx = currIdx + len;
end

end

