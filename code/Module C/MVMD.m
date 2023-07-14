clc;
clear;
close all;
warning off;
%% Load data
%load spring_angle1.mat;
dataset=xlsread('...\windpower_angle4.xlsx');
%% Setting parameters
[mm,nn]=size(dataset);
signal=dataset(:,[4,12]);
K=10;
alpha=1075;
tau=0.1;
DC=0;
init=1; % all omegas start uniformly distributed
tol=1e-7;
[u, u_hat, omega] = MVMD_ver1(signal, alpha, tau, K, DC, init, tol);
%% Plot
l=size(u);
i=l(1);
j=l(2);
k=l(3);
for ii=1:k
    m=u(:,:,ii);
    for w=0:i-1
           subplot(i,k,ii+w*5);
           jj=w+1;
           plot(1:731,m(jj,:));
           set(gca,'xtick',[]);
           axis([1,731,-inf,inf])
    end
end

%% Reshape modes
u1=u(1,:,:);
s1=reshape(u1,j,k);
u2=u(2,:,:);
s2=reshape(u2,j,k);
u3=u(3,:,:);
s3=reshape(u3,j,k);
u4=u(4,:,:);
s4=reshape(u4,j,k);
u5=u(5,:,:);
s5=reshape(u5,j,k);
u6=u(6,:,:);
s6=reshape(u6,j,k);
u7=u(7,:,:);
s7=reshape(u7,j,k);
u8=u(8,:,:);
s8=reshape(u8,j,k);
u9=u(9,:,:);
s9=reshape(u9,j,k);
u10=u(10,:,:);
s10=reshape(u10,j,k);
% u11=u(11,:,:);
% s11=reshape(u11,j,k);
% u12=u(12,:,:);
% s12=reshape(u12,j,k);
% u13=u(13,:,:);
% s13=reshape(u13,j,k);
% u14=u(14,:,:);
% s14=reshape(u14,j,k);
% u15=u(15,:,:);
% s15=reshape(u15,j,k);
xlswrite('...\windpower_angle4+.xlsx',s1,1)
xlswrite('...\windpower_angle4+.xlsx',s2,2)
xlswrite('...\windpower_angle4+.xlsx',s3,3)
xlswrite('...\windpower_angle4+.xlsx',s4,4)
xlswrite('...\windpower_angle4+.xlsx',s5,5)
xlswrite('...\windpower_angle4+.xlsx',s6,6)
xlswrite('...\windpower_angle4+.xlsx',s7,7)
xlswrite('...\windpower_angle4+.xlsx',s8,8)
xlswrite('...\windpower_angle4+.xlsx',s9,9)
xlswrite('...\windpower_angle4+.xlsx',s10,10)
% xlswrite('...\windpower_angle4.xlsx',s11,11)
% xlswrite('...\windpower_angle4.xlsx',s12,12)
% xlswrite('...\windpower_angle4.xlsx',s13,13)
% xlswrite('...\windpower_angle1.xlsx',s14,14)
% xlswrite('...\windpower_angle1.xlsx',s15,15)