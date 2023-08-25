clearvars

% raalgo = 'CSMA';
raalgo = 'slottedaloha';
per = 0.1;
nodes = 50;
velocity = 100;
step = 20;

% mat_prop = readmatrix(['temp_result_', raalgo, '_prop_', num2str(per), '_', num2str(nodes), '.csv']);
% mat_forward = readmatrix(['temp_result_', raalgo, '_forward_', num2str(per), '_', num2str(nodes), '.csv']);
% mat_sred = readmatrix(['temp_result_', raalgo, '_sred_', num2str(per), '_', num2str(nodes), '.csv']);
% mat_random = readmatrix(['temp_result_', raalgo, '_random_', num2str(per), '_', num2str(nodes), '.csv']);

mat_slot_10_5 = readmatrix(['result_slottedaloha_prop_', num2str(per), '_', num2str(nodes), '_10_5.csv']);
mat_slot_10_20 = readmatrix(['result_slottedaloha_prop_', num2str(per), '_', num2str(nodes), '_10_20.csv']);
mat_slot_100_5 = readmatrix(['result_slottedaloha_prop_', num2str(per), '_', num2str(nodes), '_100_5.csv']);
mat_slot_100_10 = readmatrix(['result_slottedaloha_prop_', num2str(per), '_', num2str(nodes), '_100_10.csv']);
mat_slot_100_20 = readmatrix(['result_slottedaloha_prop_', num2str(per), '_', num2str(nodes), '_100_20.csv']);
mat_csma_10_5 = readmatrix(['result_CSMA_prop_', num2str(per), '_', num2str(nodes), '_10_5.csv']);
mat_csma_10_20 = readmatrix(['result_CSMA_prop_', num2str(per), '_', num2str(nodes), '_10_20.csv']);
mat_csma_100_5 = readmatrix(['result_CSMA_prop_', num2str(per), '_', num2str(nodes), '_100_5.csv']);
mat_csma_100_20 = readmatrix(['result_CSMA_prop_', num2str(per), '_', num2str(nodes), '_100_20.csv']);

mat_forward = readmatrix(['result_', raalgo, '_forward_', num2str(per), '_', num2str(nodes), '.csv']);
mat_sred = readmatrix(['result_', raalgo, '_sred_', num2str(per), '_', num2str(nodes), '.csv']);
mat_rlaqm = readmatrix(['result_', raalgo, '_rlaqm_', num2str(per), '_', num2str(nodes), '.csv']);
mat_random = readmatrix(['result_', raalgo, '_random_', num2str(per), '_', num2str(nodes), '.csv']);


%% Matrix slicing
% slicing = 1000;
% len = size(mat_prop_100_5, 1);
% mat_prop_100_5 = mat_prop_100_5(len-slicing+1:1:end, :);
% mat_forward = mat_forward(len-slicing+1:1:end, :);
% mat_sred = mat_sred(len-slicing+1:1:end, :);
% mat_rlaqm = mat_rlaqm(len-slicing+1:1:end, :);
% mat_random = mat_random(len-slicing+1:1:end, :);

%% MAC protocol comparison
% Fig. 9.

slicing = 1000;
score_slot_100_5 = reshape(mat_slot_100_5(:,7), [slicing, 10]);
[~, imax_score_slot_100_5] = max(mean(score_slot_100_5));
[~, imin_score_slot_100_5] = min(mean(score_slot_100_5));
score_slot_100_20 = reshape(mat_slot_100_20(:,7), [slicing, 10]);
[~, imax_score_slot_100_20] = max(mean(score_slot_100_20));
[~, imin_score_slot_100_20] = min(mean(score_slot_100_20));
score_csma_100_5 = reshape(mat_csma_100_5(:,7), [slicing, 10]);
[~, imax_score_csma_100_5] = max(mean(score_csma_100_5));
[~, imin_score_csma_100_5] = min(mean(score_csma_100_5));
score_csma_100_20 = reshape(mat_csma_100_20(:,7), [slicing, 10]);
[~, imax_score_csma_100_20] = max(mean(score_csma_100_20));
[~, imin_score_csma_100_20] = min(mean(score_csma_100_20));


x = 1:1:slicing;
figure()
hold on
fill([x flip(x)],[score_slot_100_5(:,imin_score_slot_100_5)' flip(score_slot_100_5(:,imax_score_slot_100_5))'], 'r', 'FaceAlpha', 0.15, 'LineStyle', "none")
fill([x flip(x)],[score_slot_100_20(:,imin_score_slot_100_20)' flip(score_slot_100_20(:,imax_score_slot_100_20))'], 'g', 'FaceAlpha', 0.15, 'LineStyle', "none")
fill([x flip(x)],[score_csma_100_5(:,imin_score_csma_100_5)' flip(score_csma_100_5(:,imax_score_csma_100_5))'], 'b', 'FaceAlpha', 0.15, 'LineStyle', "none")
fill([x flip(x)],[score_csma_100_20(:,imin_score_csma_100_20)' flip(score_csma_100_20(:,imax_score_csma_100_20))'], 'c', 'FaceAlpha', 0.15, 'LineStyle', "none")


plot(x, mean(score_slot_100_5, 2), 'r', 'LineWidth', 2)
plot(x, mean(score_slot_100_20, 2), 'g', 'LineWidth', 2)
plot(x, mean(score_csma_100_5, 2), 'b', 'LineWidth', 2)
plot(x, mean(score_csma_100_20, 2), 'c', 'LineWidth', 2)

legend('', '', '', '', 'Slotted ALOHA, 100km/s, 5ms', 'Slotted ALOHA, 100km/s, 20ms', 'CSMA, 100km/s, 5ms', 'CSMA, 100km/s, 20ms', 'Location', 'best')
xlabel('Episode #')
ylabel('Cumulative rewards')
title('')
grid on

%% Reward evolutions
% 속도가 높으면 채널이 쉽게 바뀐다 -> 
% Fig. 12.

slicing = 1000;
score_prop_10_5 = reshape(mat_slot_10_5(:,7), [slicing, 10]);
[~, imax_score_prop_10_5] = max(mean(score_prop_10_5));
[~, imin_score_prop_10_5] = min(mean(score_prop_10_5));
score_prop_10_20 = reshape(mat_slot_10_20(:,7), [slicing, 10]);
[~, imax_score_prop_10_20] = max(mean(score_prop_10_20));
[~, imin_score_prop_10_20] = min(mean(score_prop_10_20));
score_slot_100_5 = reshape(mat_slot_100_5(:,7), [slicing, 10]);
[~, imax_score_prop_100_5] = max(mean(score_slot_100_5));
[~, imin_score_prop_100_5] = min(mean(score_slot_100_5));
score_slot_100_20 = reshape(mat_slot_100_20(:,7), [slicing, 10]);
[~, imax_score_prop_100_20] = max(mean(score_slot_100_20));
[~, imin_score_prop_100_20] = min(mean(score_slot_100_20));
% step = 10;

x = 1:1:slicing;
figure()
hold on
fill([x flip(x)],[score_prop_10_5(:,imin_score_prop_10_5)' flip(score_prop_10_5(:,imax_score_prop_10_5))'], 'r', 'FaceAlpha', 0.15, 'LineStyle', "none")
fill([x flip(x)],[score_prop_10_20(:,imin_score_prop_10_20)' flip(score_prop_10_20(:,imax_score_prop_10_20))'], 'g', 'FaceAlpha', 0.15, 'LineStyle', "none")
fill([x flip(x)],[score_slot_100_5(:,imin_score_prop_100_5)' flip(score_slot_100_5(:,imax_score_prop_100_5))'], 'b', 'FaceAlpha', 0.15, 'LineStyle', "none")
fill([x flip(x)],[score_slot_100_20(:,imin_score_prop_100_20)' flip(score_slot_100_20(:,imax_score_prop_100_20))'], 'c', 'FaceAlpha', 0.15, 'LineStyle', "none")
plot(x, mean(score_prop_10_5, 2), 'r', 'LineWidth', 2)
plot(x, mean(score_prop_10_20, 2), 'g', 'LineWidth', 2)
plot(x, mean(score_slot_100_5, 2), 'b', 'LineWidth', 2)
plot(x, mean(score_slot_100_20, 2), 'c', 'LineWidth', 2)

legend('', '', '', '', '10km/s, 5ms', '10km/s, 20ms', '100km/s, 5ms', '100km/s, 20ms', 'Location', 'best')
xlabel('Episode #')
ylabel('Cumulative rewards')
title('')
grid on


%% Cumulative rewards
% Fig. 5.

slicing = 1000;
score_slot_100_5 = reshape(mat_slot_100_5(:,7), [slicing, 10]);
score_slot_100_10 = reshape(mat_slot_100_10(:,7), [slicing, 10]);
score_slot_100_20 = reshape(mat_slot_100_20(:,7), [slicing, 10]);
score_forward = reshape(mat_forward(:,7), [slicing, 10]);
score_sred = reshape(mat_sred(:,7), [slicing, 10]);
score_rlaqm = reshape(mat_rlaqm(:,7), [slicing, 10]);
score_random = reshape(mat_random(:,7), [slicing, 10]);
step = 20;

x = 1:step:slicing;
% p1 = plot(x, score_prop(1:step:end), '-o');
% p2 = plot(x, score_forward(1:step:end), '-^');
% p3 = plot(x, score_sred(1:step:end), '-*');
y1 = smoothdata(mean(score_slot_100_5, 2), 'movmean', step);
y2 = smoothdata(mean(score_slot_100_10, 2), 'movmean', step);
% y3 = smoothdata(mean(score_forward, 2), 'movmean', step);
y4 = smoothdata(mean(score_sred, 2), 'movmean', step);
y5 = smoothdata(mean(score_rlaqm, 2), 'movmean', step);
% y6 = smoothdata(mean(score_random, 2), 'movmean', step);

% fill([x flip(x)],[score_prop_5(:,imin_score_prop_5)' flip(score_prop_5(:,imax_score_prop_5))'], 'r', 'FaceAlpha', 0.15)
% fill([x flip(x)],[score_prop_20(:,imin_score_prop_20)' flip(score_prop_20(:,imax_score_prop_20))'], 'g', 'FaceAlpha', 0.15)
% fill([x flip(x)],[score_forward(:,imin_score_forward)' flip(score_forward(:,imax_score_forward))'], 'b', 'FaceAlpha', 0.15)
% fill([x flip(x)],[score_sred(:,imin_score_sred)' flip(score_sred(:,imax_score_sred))'], 'c', 'FaceAlpha', 0.15)
% fill([x flip(x)],[score_rlaqm(:,imin_score_rlaqm)' flip(score_rlaqm(:,imax_score_rlaqm))'], 'm', 'FaceAlpha', 0.15)
% fill([x flip(x)],[score_random(:,imin_score_random)' flip(score_random(:,imax_score_random))'], 'y', 'FaceAlpha', 0.15)
% plot(x, mean(score_prop_5, 2), 'LineWidth', 2)
% plot(x, mean(score_prop_20, 2), 'LineWidth', 2)
% plot(x, mean(score_forward, 2), 'LineWidth', 2)
% plot(x, mean(score_sred, 2), 'LineWidth', 2)
% plot(x, mean(score_rlaqm, 2), 'LineWidth', 2)
% plot(x, mean(score_random, 2), 'LineWidth', 2)

figure()
hold on
p1 = plot(x, y1(1:step:end), '-s');
p2 = plot(x, y2(1:step:end), '-^');
% p3 = plot(x, y3(1:step:end), '-*');
p4 = plot(x, y4(1:step:end), '-o');
p5 = plot(x, y5(1:step:end), '-d');
% p6 = plot(x, y5(1:step:end), '-x');
p1.LineWidth = 1; p2.LineWidth = 1; p4.LineWidth = 1; p5.LineWidth = 1;
legend('DeepAAQM w. 5ms target', 'DeepAAQM w. 20ms target', 'SRED', 'RL-AQM', 'Location', 'best')
% axis([1 slicing min([y1(1:step:end); y2(1:step:end); y3(1:step:end); y4(1:step:end); y5(1:step:end); y6(1:step:end)]) max([y1(1:step:end); y2(1:step:end); y3(1:step:end); y4(1:step:end); y5(1:step:end); y6(1:step:end)])]);
xlabel('Episode #')
ylabel('Cumulative rewards')
title('')
grid on



%% Velocity variation



%% Channel access method variation


%% Mean AoI value
% Fig. 7.

meanaoi_prop_5 = reshape(mat_slot_100_5(:,8), [slicing, 10]);
meanaoi_prop_20 = reshape(mat_slot_100_20(:,8), [slicing, 10]);
meanaoi_forward = reshape(mat_forward(:,8), [slicing, 10]);
meanaoi_sred = reshape(mat_sred(:,8), [slicing, 10]);
meanaoi_rlaqm = reshape(mat_rlaqm(:,8), [slicing, 10]);
meanaoi_random = reshape(mat_random(:,8), [slicing, 10]);

figure()
hold on
x = 1:step:slicing;
% p1 = plot(x, score_prop(1:step:end), '-o');
% p2 = plot(x, score_forward(1:step:end), '-^');
% p3 = plot(x, score_sred(1:step:end), '-*');
y1 = smoothdata(mean(meanaoi_prop_5, 2), 'movmean', step);
y2 = smoothdata(mean(meanaoi_prop_20, 2), 'movmean', step);
% y3 = smoothdata(mean(meanaoi_forward, 2), 'movmean', step);
y4 = smoothdata(mean(meanaoi_sred, 2), 'movmean', step);
y5 = smoothdata(mean(meanaoi_rlaqm, 2), 'movmean', step);
% y6 = smoothdata(meanaoi_random, 'movmean', step);
p1 = plot(x, y1(1:step:end), '-s');
p2 = plot(x, y2(1:step:end), '-^');
% p3 = plot(x, y3(1:step:end), '-*');
p4 = plot(x, y4(1:step:end), '-o');
p5 = plot(x, y5(1:step:end), '-d');

p1.LineWidth = 1; p2.LineWidth = 1; p4.LineWidth = 1; p5.LineWidth = 1;
legend('DeepAAQM w. 5ms target', 'DeepAAQM w. 20ms target', 'SRED', 'RL-AQM', 'Location', 'best')
axis([1 slicing min([y1(1:step:end); y2(1:step:end); y4(1:step:end); y5(1:step:end)]) max([y1(1:step:end); y2(1:step:end); y4(1:step:end); y5(1:step:end)])]);
xlabel('Episode #')
ylabel('Mean AoI (ms)')
title('')
grid on

%% Maximum peak AoI value
% Fig. 6.

peakaoi_prop_5 = reshape(mat_slot_100_5(:,9), [slicing, 10]);
peakaoi_prop_10 = reshape(mat_slot_100_10(:,9), [slicing, 10]);
peakaoi_prop_20 = reshape(mat_slot_100_20(:,9), [slicing, 10]);
peakaoi_forward = reshape(mat_forward(:,9), [slicing, 10]);
peakaoi_sred = reshape(mat_sred(:,9), [slicing, 10]);
peakaoi_rlaqm = reshape(mat_rlaqm(:,9), [slicing, 10]);

figure()
hold on
x = 1:step:slicing;
% p1 = plot(x, score_prop(1:step:end), '-o');
% p2 = plot(x, score_forward(1:step:end), '-^');
% p3 = plot(x, score_sred(1:step:end), '-*');
y1 = smoothdata(mean(peakaoi_prop_5, 2), 'movmean', step);
y2 = smoothdata(mean(peakaoi_prop_20, 2), 'movmean', step);
% y3 = smoothdata(mean(peakaoi_forward, 2), 'movmean', step);
y4 = smoothdata(mean(peakaoi_sred, 2), 'movmean', step);
y5 = smoothdata(mean(peakaoi_rlaqm, 2), 'movmean', step);

p1 = plot(x, y1(1:step:end), '-s');
p2 = plot(x, y2(1:step:end), '-^');
% p3 = plot(x, y3(1:step:end), '-*');
p4 = plot(x, y4(1:step:end), '-o');
p5 = plot(x, y5(1:step:end), '-d');
p1.LineWidth = 1; p2.LineWidth = 1; p4.LineWidth = 1; p5.LineWidth = 1;
legend('DeepAAQM w. 5ms target', 'DeepAAQM w. 20ms target', 'SRED', 'RL-AQM', 'Location', 'best')
axis([1 slicing min([y1(1:step:end); y2(1:step:end); y4(1:step:end); y5(1:step:end)]) max([y1(1:step:end); y2(1:step:end); y4(1:step:end); y5(1:step:end)])]);
xlabel('Episode #')
ylabel('Peak AoI (ms)')
title('')
grid on

%% Avg. consumed power
% Fig. 8.

pow_prop_5 = reshape(mat_slot_100_5(:,end), [slicing, 10]);
pow_prop_20 = reshape(mat_slot_100_20(:,end), [slicing, 10]);
pow_forward = reshape(mat_forward(:,end), [slicing, 10]);
pow_sred = reshape(mat_sred(:,end), [slicing, 10]);
pow_rlaqm = reshape(mat_rlaqm(:,end), [slicing, 10]);
% pow_random = reshape(mat_random(:,end), [slicing, 10]);


figure()
hold on
x = 1:step:slicing;
% p1 = plot(x, score_prop(1:step:end), '-o');
% p2 = plot(x, score_forward(1:step:end), '-^');
% p3 = plot(x, score_sred(1:step:end), '-*');
y1 = smoothdata(mean(pow_prop_5, 2), 'movmean', step);
y2 = smoothdata(mean(pow_prop_20, 2), 'movmean', step);
% y3 = smoothdata(mean(pow_forward, 2), 'movmean', step);
y4 = smoothdata(mean(pow_sred, 2), 'movmean', step);
y5 = smoothdata(mean(pow_rlaqm, 2), 'movmean', step);
% y5 = smoothdata(mean(pow_random, 2), 'movmean', step);
p1 = plot(x, y1(1:step:end), '-s');
p2 = plot(x, y2(1:step:end), '-^');
% p3 = plot(x, y3(1:step:end), '-*');
p4 = plot(x, y4(1:step:end), '-o');
p5 = plot(x, y5(1:step:end), '-d');
p1.LineWidth = 1; p2.LineWidth = 1; p4.LineWidth = 1; p5.LineWidth = 1;
legend('DeepAAQM w. 5ms target', 'DeepAAQM w. 20ms target', 'SRED', 'RL-AQM', 'Location', 'best')
axis([1 slicing min([y1(1:step:end); y2(1:step:end); y4(1:step:end); y5(1:step:end)]) max([y1(1:step:end); y2(1:step:end); y4(1:step:end); y5(1:step:end)])]);
xlabel('Episode #')
ylabel('Consumed Power (Watts)')
title('')
grid on

%% MeanAoI CDF
% aoi_prop = mat_prop_100_5(:,7);
% aoi_forward = mat_forward(:,7);
% aoi_sred = mat_sred(:,7);
% aoi_rlaqm = mat_rlaqm(:,7);


figure()
hold on
p1 = cdfplot(mat_slot_100_5(:,8));
p2 = cdfplot(mat_slot_100_20(:,8));
p3 = cdfplot(mat_forward(:,8));
p4 = cdfplot(mat_sred(:,8));
p5 = cdfplot(mat_rlaqm(:,8));
% p5 = cdfplot(aoi_random);
p1.LineWidth = 2; p2.LineWidth = 2; p3.LineWidth = 2; p4.LineWidth = 2; p5.LineWidth = 2;
% p1.LineStyle = '-'; p2.LineStyle = '--'; p3.LineStyle = '-.'; p4.LineStyle = ':';
legend('DeepAAQM w. 5ms target', 'DeepAAQM w. 20ms target', 'Store-and-forward', 'SRED', 'RL-AQM', 'Location', 'best')
% axis([0 max([aoi_prop; aoi_forward; aoi_sred; aoi_rlaqm]) 0 1]);
xlabel('Mean Delay (ms)')
ylabel('Empirical CDF')
title('')

%% peak AoI CDF
% max_aoi_prop = mat_prop_100_5(:,8);
% max_aoi_forward = mat_forward(:,8);
% max_aoi_sred = mat_sred(:,8);
% max_aoi_rlaqm = mat_rlaqm(:,8);
% max_aoi_random = mat_random(:,8);

figure()
hold on
p1 = cdfplot(mat_slot_100_5(:,9));
p2 = cdfplot(mat_slot_100_20(:,9));
p3 = cdfplot(mat_forward(:,9));
p4 = cdfplot(mat_sred(:,9));
p5 = cdfplot(mat_rlaqm(:,9));
% p5 = cdfplot(max_aoi_random);
p1.LineWidth = 2; p2.LineWidth = 2; p3.LineWidth = 2; p4.LineWidth = 2; p5.LineWidth = 2;
% p1.LineStyle = '-'; p2.LineStyle = '--'; p3.LineStyle = '-.'; p4.LineStyle = ':';
legend('DeepAAQM w. 5ms target', 'DeepAAQM w. 20ms target', 'Store-and-forward', 'SRED', 'RL-AQM', 'Location', 'best')
% axis([0 max([aoi_prop; aoi_forward; aoi_sred; aoi_rlaqm]) 0 1]);
xlabel('Peak Age of Information (ms)')
ylabel('Empirical CDF')
title('')

%% Avg. consumed power CDF

figure()
hold on
p1 = cdfplot(mat_slot_100_5(:,end));
p2 = cdfplot(mat_slot_100_20(:,end));
p3 = cdfplot(mat_forward(:,end));
p4 = cdfplot(mat_sred(:,end));
p5 = cdfplot(mat_rlaqm(:,end));

p1.LineWidth = 2; p2.LineWidth = 2; p3.LineWidth = 2; p4.LineWidth = 2; p5.LineWidth = 2;
legend('DeepAAQM w. 5ms target', 'DeepAAQM w. 20ms target', 'Store-and-forward', 'SRED', 'RL-AQM', 'Location', 'best')
% axis([0 max([pow_prop; pow_forward; pow_sred; pow_rlaqm]) 0 1]);
xlabel('Average consumed power (Watts)')
ylabel('Empirical CDF')
title('')

%% Node number comparisons
% CSMA가 노드의 수에 더 다이나믹하게 반응하기 때문에 CSMA로 하였음.
% 센서 노드 수가 적으면 수집한 데이터 양 자체가 적어서 AoI를 맞추기 힘듦.
% 센서 노드 수가 너무 많으면 collision때문에 AoI를 맞추기 힘듦.
% Harsh condition을 보기 위해 tau_target = 5ms.
% AoI value requirement meet하기 위해서는 적절한 수의 센서 노드가 필요하나 out of scope. 나중에
% 다뤄보겠음.

mat_node_10 = readmatrix('result_CSMA_prop_0.1_10_100_5.csv');
mat_node_50 = readmatrix('result_CSMA_prop_0.1_50_100_5.csv');
mat_node_100 = readmatrix('result_CSMA_prop_0.1_100_100_5.csv');

reward_10 = reshape(mat_node_10(:,7), [1000, 10]);
reward_50 = reshape(mat_node_50(:,7), [1000, 10]);
reward_100 = reshape(mat_node_100(:,7), [1000, 10]);
reward_10 = mean(reward_10(end,:));
reward_50 = mean(reward_50(end,:));
reward_100 = mean(reward_100(end,:));

pow_10 = reshape(mat_node_10(:,end), [1000, 10]);
pow_50 = reshape(mat_node_50(:,end), [1000, 10]);
pow_100 = reshape(mat_node_100(:,end), [1000, 10]);
pow_10 = mean(pow_10(end,:));
pow_50 = mean(pow_50(end,:));
pow_100 = mean(pow_100(end,:));

figure()
hold on

plot([1 2 3], [reward_10 reward_50 reward_100])
plot([1 2 3], 100*[pow_10 pow_50 pow_100])
legend('Reward', 'Power', 'Location', 'Best')

%% Fig. 10.
slicing = 1000;
mat_node_10 = readmatrix('result_slottedaloha_prop_0.1_10_100_5.csv');
mat_node_50 = readmatrix('result_slottedaloha_prop_0.1_50_100_5.csv');
mat_node_100 = readmatrix('result_slottedaloha_prop_0.1_100_100_5.csv');

score_node_10 = reshape(mat_node_10(:,7), [slicing, 10]);
[~, imax_score_node_10] = max(mean(score_node_10));
[~, imin_score_node_10] = min(mean(score_node_10));
score_node_50 = reshape(mat_node_50(:,7), [slicing, 10]);
[~, imax_score_node_50] = max(mean(score_node_50));
[~, imin_score_node_50] = min(mean(score_node_50));
score_node_100 = reshape(mat_node_100(:,7), [slicing, 10]);
[~, imax_score_node_100] = max(mean(score_node_100));
[~, imin_score_node_100] = min(mean(score_node_100));

x = 1:1:slicing;
figure()
hold on
fill([x flip(x)],[score_node_10(:,imin_score_node_10)' flip(score_node_10(:,imax_score_node_10))'], 'r', 'FaceAlpha', 0.15, 'LineStyle', "none")
fill([x flip(x)],[score_node_50(:,imin_score_node_50)' flip(score_node_50(:,imax_score_node_50))'], 'g', 'FaceAlpha', 0.15, 'LineStyle', "none")
fill([x flip(x)],[score_node_100(:,imin_score_node_100)' flip(score_node_100(:,imax_score_node_100))'], 'b', 'FaceAlpha', 0.15, 'LineStyle', "none")
plot(x, mean(score_node_10, 2), 'r', 'LineWidth', 1)
plot(x, mean(score_node_50, 2), 'g', 'LineWidth', 1)
plot(x, mean(score_node_100, 2), 'b', 'LineWidth', 1)

legend('', '', '', 'Number of CMs: 10 w. 5ms target', 'Number of CMs: 50 w. 5ms target', 'Number of CMs: 100 w. 5ms target', 'Location', 'best')
xlabel('Episode #')
ylabel('Cumulative rewards')
title('')
grid on

%% Fig. 11.
slicing = 1000;
mat_node_10 = readmatrix('result_CSMA_prop_0.1_10_100_5.csv');
mat_node_50 = readmatrix('result_CSMA_prop_0.1_50_100_5.csv');
mat_node_100 = readmatrix('result_CSMA_prop_0.1_100_100_5.csv');

score_node_10 = reshape(mat_node_10(:,7), [slicing, 10]);
[~, imax_score_node_10] = max(mean(score_node_10));
[~, imin_score_node_10] = min(mean(score_node_10));
score_node_50 = reshape(mat_node_50(:,7), [slicing, 10]);
[~, imax_score_node_50] = max(mean(score_node_50));
[~, imin_score_node_50] = min(mean(score_node_50));
score_node_100 = reshape(mat_node_100(:,7), [slicing, 10]);
[~, imax_score_node_100] = max(mean(score_node_100));
[~, imin_score_node_100] = min(mean(score_node_100));

x = 1:1:slicing;
figure()
hold on
fill([x flip(x)],[score_node_10(:,imin_score_node_10)' flip(score_node_10(:,imax_score_node_10))'], 'r', 'FaceAlpha', 0.15, 'LineStyle', "none")
fill([x flip(x)],[score_node_50(:,imin_score_node_50)' flip(score_node_50(:,imax_score_node_50))'], 'g', 'FaceAlpha', 0.15, 'LineStyle', "none")
fill([x flip(x)],[score_node_100(:,imin_score_node_100)' flip(score_node_100(:,imax_score_node_100))'], 'b', 'FaceAlpha', 0.15, 'LineStyle', "none")
plot(x, mean(score_node_10, 2), 'r', 'LineWidth', 1)
plot(x, mean(score_node_50, 2), 'g', 'LineWidth', 1)
plot(x, mean(score_node_100, 2), 'b', 'LineWidth', 1)

legend('', '', '', 'Number of CMs: 10 w. 5ms target', 'Number of CMs: 50 w. 5ms target', 'Number of CMs: 100 w. 5ms target', 'Location', 'best')
xlabel('Episode #')
ylabel('Cumulative rewards')
title('')
grid on