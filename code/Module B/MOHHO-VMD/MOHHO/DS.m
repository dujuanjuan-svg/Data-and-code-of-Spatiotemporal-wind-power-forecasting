function Ds = DS(output_test,T_sim)
% input:
%      output_test: Testing data 1xN
%      T_sim:       simulation testing data  1xN

m = size(T_sim,2);
data = [output_test;T_sim];

for j = 2:m
    if (data(1,j)-data(1,j-1))*(data(2,j)-data(2,j-1)) >= 0
        d(1,j-1) = 1;
    else
        d(1,j-1) = 0;
    end
end
Ds = sum(d(1,:)) / m;

end

