function [z, Sol] = MVMD(x)
%% Load data
dataset=xlsread('...\windpower_angle1.xlsx');
%% Setting parameters
[mm,nn]=size(dataset);
signal=dataset(:,[4,12]);
DC=0;
init=1; % all omegas start uniformly distributed
tol=1e-7;
x(3)=round(x(3));
[u, ~, ~] = MVMD_ver1(signal, x(1), x(2), x(3), DC, init, tol);
%% Plot
l=size(u);
i=l(1);
j=l(2);
k=l(3);
% for ii=1:k
%     m=u(:,:,ii);
%     for w=0:i-1
%            subplot(i,k,ii+w*5);
%            jj=w+1;
%            plot(1:731,m(jj,:));
%            set(gca,'xtick',[]);
%            axis([1,731,-inf,inf])
%     end
% end
%% Loss functions
% The range entropy of the modes
x1=zeros(k,i);
for i1=1:k
k1=u(:,:,i1);
k1=k1';
[v1,g1]=size(k1);
    for z1=1:g1
        data=k1(:,z1);
        y1=RangeEn_B(data,5,0.8);
        x1(i1,z1)=y1;
    end
end
f1=sum(sum(x1))./(k*i);

%The entropy of the error component
x2=zeros(k,j);
 for i2=1:k
k2=u(:,:,i2);
[p2,q2]=size(k2);
    for t2=1:q2
        sum1=sum(k2(:,t2));
        x2(i2,t2)=sum1;
    end
 end
x2=x2';
sum3=signal(:,:)-x2(:,:);
sum5=zeros(1,k);
for i3=1:k
data=sum3(:,i3);
sum4=RangeEn_B(data,5,0.8);
sum5(i3)=sum4;
end
f2=sum(sum5)/k;

% Adjacent Pearson correlation
x4=zeros(k,i-1);
for i5=1:k
k4=u(:,:,i5);
[m4,n4]=size(k4);
        for i6=1:m4-1
            coefficient=corrcoef(k4(i6,:),k4(i6+1,:));
            x4(i5,i6)=coefficient(1,2);
        end
end
x4=x4';
f3=sum(sum(x4))./(k*(i-1));

% Adjacent maximum orthogonality
max_x4=max(x4);
max_x5=max(max_x4,[],2);
f4=max_x5;

z=[f1 
   f2 
   f3
   f4];
   Sol.f1=f1;
   Sol.f2=f2;
   Sol.f3=f3;
   Sol.f4=f4;
   Sol.x=x;
end

