clear;
M = dlmread('/home/shouwang/lianghu/user_log_format1.csv',',',1,0);
M = M(M(:,7) == 2, [1,2,6,7]);
[~,~,users] = unique(M(:, 1), 'stable');
[~,~,items] = unique(M(:, 2), 'stable');
[~,~,dates] = unique(M(:, 3), 'stable');
testingDate = max(dates) - 30;

itemcnt = sparse(1, items, 1);
filter = itemcnt > 19;
filter = filter(items);

Mt = [users(filter), items(filter), dates(filter)];

Mt = sortrows(Mt,[1,3]);
mask = true(size(Mt,1),1);

startUser = 1;
startSess = 1;
userIdx = Mt(1,1);
dateIdx = Mt(1,3);
sessCnt = 0;
for i = 2:size(Mt,1)
    if userIdx ~= Mt(i,1) || dateIdx ~= Mt(i,3)
        sessionLen = i - startSess;
        if sessionLen < 2
            mask(startSess:i-1) = false;
        else
            sessCnt = sessCnt + 1;
        end
        startSess = i;
        dateIdx = Mt(i,3);
    end
    
    if userIdx ~= Mt(i,1)
        if sessCnt < 3
            mask(startUser:i-1) = false;
        end
        sessCnt = 0;
        startUser = i;
        userIdx = Mt(i,1);
    end
end
mask(startUser:end) = false;



Mt = Mt(mask, :);
[~,~,users] = unique(Mt(:, 1), 'stable');
[~,~,items] = unique(Mt(:, 2), 'stable');
dates = Mt(:,3);
Mt = [users, items, dates];
nUser = length(unique(users));
nItem = length(unique(items));
testingItems = false(1, nItem);
testingItems(items(dates <= testingDate)) = true;




sessIdx = 1;
userIdx = Mt(1,1);
dateIdx = Mt(1,3);
session = cell(1, ceil(size(Mt,1)/10));
session{1}.User = userIdx;
session{1}.Date = dateIdx;
sessionItem = Mt(1,2);
for i = 2:size(Mt,1)
    if userIdx ~= Mt(i,1) ||  dateIdx ~= Mt(i,3)
        session{sessIdx}.Session = sessionItem;
        sessIdx = sessIdx + 1;
        userIdx = Mt(i,1);
        dateIdx = Mt(i,3);
        session{sessIdx}.User = userIdx;
        session{sessIdx}.Date = dateIdx;
        sessionItem = Mt(i,2);
    else
        sessionItem = [sessionItem, Mt(i,2)];
    end
end
session{sessIdx}.Session = sessionItem;
session = session(1:sessIdx);

totalLen = 0;
testingIdx = false(sessIdx, 1);
userIdx = session{1}.User;
nUserSession = 0;
for i = 1:sessIdx
    if session{i}.Date > testingDate && ...
       all(testingItems(session{i}.Session))
       testingIdx(i) = rand > 0.8;
    end
    totalLen = totalLen + length(session{i}.Session);
end


testingSet = session(testingIdx);
trainingSet = session(~testingIdx);

nTest = length(testingSet);
testllo = cell(1, nTest*3);
actualCnt = 0;
for i = 1:nTest
    tuser = testingSet{i}.User;
    sess = testingSet{i}.Session;
    len = length(sess);
    for j = 1:len
        actualCnt = actualCnt + 1;
        testllo{actualCnt}.User = tuser;
        testllo{actualCnt}.Position = j;
        testllo{actualCnt}.TestCase = sess(j);
        testllo{actualCnt}.Session = sess;
        testllo{actualCnt}.Session(j) = [];
    end
end
testllo = testllo(1:actualCnt);

testLast = cell(1, nTest);
for i = 1:nTest
    sess = testingSet{i}.Session;
    testLast{i}.User = testingSet{i}.User;
    testLast{i}.Position = length(sess);
    testLast{i}.Session = sess(1:end-1);
    testLast{i}.TestCase = sess(end);
end

trainData = MakePSHMDataCases(trainingSet);