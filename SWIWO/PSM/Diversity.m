function [ diver ] = Diversity( data )
%DIVERSITY 此处显示有关此函数的摘要
%   此处显示详细说明
diver = 0;
nSess = size(data, 1);

parfor i = 1:nSess-1
    currData = data(i, :);
    for j = i+1:nSess
        diver = diver + (1 - length(intersect(data(j, :), currData)) / length(union(data(j, :), currData)));
    end
end

diver = diver / (0.5*nSess*(nSess - 1));

end

