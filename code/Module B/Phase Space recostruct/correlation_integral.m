%Correlation integral calculation
function C_I=correlation_integral(X,M,r)

sum_H=0;
for i=1:M-1
    for j=i+1:M
        d=norm((X(:,i)-X(:,j)),inf);
        if r>d    
           sita=heaviside(r,d);
           sum_H=sum_H+1;
        end
    end
end
C_I=2*sum_H/(M*(M-1));

