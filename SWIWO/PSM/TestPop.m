popItemCnt = full(sparse(1, Mt(:,4), 1, 1, nItem));
[~, Idx] = sort(popItemCnt, 'descend');
rank(Idx) = 1:nItem;
 
nTest = length(testllo);
rank_pop_llo = zeros(nTest, 1);
for i=1:nTest
    rank_pop_llo(i) = rank(testllo{i}.TestCase);
end
recallAt10_pop_LLO = mean(rank_pop_llo<=10);
recallAt20_pop_LLO = mean(rank_pop_llo<=20);
recallAt50_pop_LLO = mean(rank_pop_llo<=50);
mrr_pop_LLO = mean(1./rank_pop_llo);
auc_pop_LLO = mean( (nItem - rank_pop_llo) ./ nItem);

nTest = length(testLast);
rank_pop_Last = zeros(nTest, 1);
for i=1:nTest
    rank_pop_Last(i) = rank(testLast{i}.TestCase);
end
recallAt10_pop_Last = mean(rank_pop_Last<=10);
recallAt20_pop_Last = mean(rank_pop_Last<=20);
recallAt50_pop_Last = mean(rank_pop_Last<=50);
mrr_pop_Last = mean(1./rank_pop_Last);
auc_pop_Last = mean( (nItem - rank_pop_Last) ./ nItem);

