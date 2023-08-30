function img_eq = histogeq(original)


Bout = zeros(256,1);
[rows,columns] = size(original);
L = double(max(max(original)));
hist = histogram(original);
cdf = hist(:,4);
const = (rows.*columns) ./ L;
img_eq = zeros(rows,columns);

Bout(1:256,1) = cdf(1:256,1)./const - 1;


for i = 1:rows
    for j = 1:columns
        img_eq(i,j) = Bout(original(i,j)+1,1);
    end
end
end