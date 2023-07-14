
function Data=reconstitution(data,m,tau)

N=length(data); 
M=N-(m-1)*tau; 
Data=zeros(m,M);
for j=1:M
  for i=1:m
    Data(i,j)=data((i-1)*tau+j);
  end
end
end