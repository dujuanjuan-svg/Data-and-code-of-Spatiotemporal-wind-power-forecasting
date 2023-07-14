
clc;
clear;
close all;
warning off;
%% Load data
dataset=xlsread('...\windpower_angle4.xlsx');
%% Setting parameters
signal=dataset(:,3);
DC=0;
init=1; % all omegas start uniformly distributed
tol=1e-7;
alpha =2503;           % moderate bandwidth constraint
tau   = 0.1462;              % noise-tolerance (no strict fidelity enforcement)
K     =12;  

[u, u_hat, omega] = VMD(signal, alpha, tau, K, DC, init, tol);
%% Plot
sum_sum=sum(u);
xx=1:size(sum_sum,2);
yy=sum_sum-signal';
error_error=sum(abs(yy));
plot(xx,yy,'r')
%% save
u=u';
[row,columns]=size(u);
for i=1:columns
xlswrite('...\windpower_angle4.xlsx',u(:,i),i);
end