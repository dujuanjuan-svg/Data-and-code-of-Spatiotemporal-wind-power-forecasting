clc;
clear;
close all;
warning off;
dataset=xlsread('...\mali.xlsx');
a=dataset;
windpower=a;
% Angle division
[m,n]=size(windpower);
j1=1;
j2=1;
j3=1;
j4=1;
for i=1:m
    if 0<=windpower(i,3)& windpower(i,3)<90
        windpower_angle1(j1,:)=windpower(i,:);
        j1=j1+1;
    elseif 90<=windpower(i,3)&windpower(i,3)<180
        windpower_angle2(j2,:)=windpower(i,:);
        j2=j2+1;
    elseif 180<=windpower(i,3)&windpower(i,3)<270
        windpower_angle3(j3,:)=windpower(i,:);
        j3=j3+1;
    else 
        windpower_angle4(j4,:)=windpower(i,:);
        j4=j4+1;
    end
end
windpower_correfficient1=corr(windpower_angle1);
windpower_correfficient2=corr(windpower_angle2);
windpower_correfficient3=corr(windpower_angle3);
windpower_correfficient4=corr(windpower_angle4);
%% 
xlswrite('...\windpower_angle1.xlsx',windpower_angle1);
xlswrite('...\windpower_angle2.xlsx',windpower_angle2);
xlswrite('...\windpower_angle3.xlsx',windpower_angle3);
xlswrite('...\windpower_angle4.xlsx',windpower_angle4);