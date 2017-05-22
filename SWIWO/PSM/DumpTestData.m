function DumpTestData( filePath, testSet )
fileID = fopen(filePath,'w');
nTrain = length(testSet);
for i = 1:nTrain
    sess = testSet{i}.Session;
    sess = [sess, testSet{i}.TestCase];
    fprintf(fileID, '%u', testSet{i}.User);
    fprintf(fileID, ' %u ', sess);
    fprintf(fileID, '\n');
end
fclose(fileID);
end

