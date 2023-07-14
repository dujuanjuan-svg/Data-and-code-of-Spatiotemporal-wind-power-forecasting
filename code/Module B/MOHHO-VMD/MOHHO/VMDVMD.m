function [z, Sol] = VMDVMD(signal,x)
DC=0;
init=1; % all omegas start uniformly distributed
tol=1e-7;
x(3)=round(x(3));
[u, u_hat, omega] = VMD(signal, x(1), x(2), x(3), DC, init, tol);
[k,t]=size(u);
%% Loss functions
% The range entropy of the modes
x1=zeros(k,1);
for i1=1:k
        data=u(i1,:);
        y1=RangeEn_B(data,5,0.8);
        x1(i1,1)=y1;
end
f1=sum(x1)./k;

%The entropy of the error component
sumsum=sum(u);
sumsum=sumsum';
difference=signal(:,:)-sumsum(:,:);
f2=RangeEn_B(difference,5,0.8);

% Adjacent Pearson correlation
x4=zeros(k-1,1);
for i6=1:k-1
    coefficient=corrcoef(u(i6,:),u(i6+1,:));
    x4(i6,1)=coefficient(1,2);
end
f3=sum(x4)./(k-1);

% Adjacent maximum orthogonality
max_x4=max(x4);
f4=max_x4;

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

