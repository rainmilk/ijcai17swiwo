function createfigure(ymatrix1)
%CREATEFIGURE(YMATRIX1)
%  YMATRIX1:  bar 矩阵数据

%  由 MATLAB 于 19-Feb-2017 23:01:56 自动生成

% 创建 figure
figure;

% 创建 axes
axes1 = axes;
hold(axes1,'on');

% 使用 bar 的矩阵输入创建多行
bar1 = bar(ymatrix1);
set(bar1(2),'DisplayName','DIV@10',...
    'FaceColor',[0 0.447058826684952 0.74117648601532]);
set(bar1(1),'DisplayName','F1@10','FaceColor',[1 0.843137264251709 0]);

% 创建 title
title('\it{LAST}');

box(axes1,'on');
% 设置其余坐标轴属性
set(axes1,'FontSize',16,'XGrid','on','XTick',[1 2 3 4 5 6],'XTickLabel',...
    {'\it{POP}','\it{FPME}','\it{PRME}','\it{GRU4Rec}','\it{SWIWO-I}','\it{SWIWO}'},...
    'YGrid','on');
% 创建 legend
legend1 = legend(axes1,'show');
set(legend1,'Location','northwest');

