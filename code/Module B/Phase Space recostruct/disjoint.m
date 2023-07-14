
function Data=disjoint(data,t)

%Length of data
for i=1:t
    for j=1:(N/t)
        Data(j,i)=data(i+(j-1)*t);
    end
end
