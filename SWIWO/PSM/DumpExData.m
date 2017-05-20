fileID = fopen('user_idx_list.txt','w');
fprintf(fileID, 'UserIdx\n');
fprintf(fileID, '%u\n', 1:nUser);
fclose(fileID);

fileID = fopen('item_idx_list.txt','w');
fprintf(fileID, 'ItemIdx\n');
fprintf(fileID, '%u\n', 1:nItem);
fclose(fileID);

fileID = fopen('trainFPMC.txt','w');
nTrain = length(trainingSet);
for i = 1:nTrain
    len = length(trainingSet{i}.Session);
    sess = trainingSet{i}.Session;
    fprintf(fileID, '%u', trainingSet{i}.User);
    fprintf(fileID, ' %u ', sess);
    fprintf(fileID, '\n');
end
fclose(fileID);

DumpTestData('testFPMC_LLO.txt', testllo);
DumpTestData('testFPMC_LAST.txt', testLast);

fileID = fopen('trainRNN.txt','w');
fprintf(fileID, 'SessionId\tItemId\tTime');
nTrain = length(trainingSet);
for i = 1:nTrain
    len = length(trainingSet{i}.Session);
    sess = trainingSet{i}.Session;
    for j = 1:len
        fprintf(fileID, '\n%u\t%u\t%u', i, sess(j), j);
    end
end
fclose(fileID);

fileID = fopen('testRNN_LLO.txt','w');
fprintf(fileID, 'SessionId\tItemId\tTime');
nTest = length(testingSet);
for i = 1:nTest
    sess = testingSet{i}.Session;
    len = length(testingSet{i}.Session);

    for j = 1:len
        fprintf(fileID, '\n%u\t%u\t%u', i, sess(j), j);
    end
end
fclose(fileID);

fileID = fopen('testRNN_LAST.txt','w');
fprintf(fileID, 'SessionId\tItemId\tTime');
nTest = length(testingSet);
for i = 1:nTest
    sess = testingSet{i}.Session;
    len = length(testingSet{i}.Session);

    fprintf(fileID, '\n%u\t%u\t%u', i, sess(end-1), j);
    fprintf(fileID, '\n%u\t%u\t%u', i, sess(end), j);
end
fclose(fileID);

fileID = fopen('trainPRME.txt','w');
nTrain = length(trainingSet);
for i = 1:nTrain
    len = length(trainingSet{i}.Session);
    sess = trainingSet{i}.Session;
    for j=2:len
        fprintf(fileID, '1\t%u\t%u\t%u\n', trainingSet{i}.User, sess(j-1), sess(j));
    end
end
fclose(fileID);

fileID = fopen('testPRME_LLO.txt','w');
nTest = length(testingSet);
for i = 1:nTest
    sess = testingSet{i}.Session;
    len = length(testingSet{i}.Session);
    for j=2:len
        fprintf(fileID, '1\t%u\t%u\t%u\n', testingSet{i}.User, sess(j-1), sess(j));
    end
end
fclose(fileID);

fileID = fopen('testPRME_Last.txt','w');
nTest = length(testingSet);
for i = 1:nTest
    sess = testingSet{i}.Session;
    len = length(testingSet{i}.Session);
    fprintf(fileID, '1\t%u\t%u\t%u\n', testingSet{i}.User, sess(end-1), sess(end));
end
fclose(fileID);