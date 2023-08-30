function histg = histogram(x)

histg = zeros(256,4); %4 columns

histg(:,1) = 0:255;

for b=1:numel(x)
    histg(x(b)+1,2) = histg(x(b)+1,2)+1; %histogram
end

histg(:,3) = histg(:,2)/sum(histg(:,2)); %Hist normalized
histg(:,4) = cumsum(histg(:,3));  %returns the cumulative sum of A starting at the beginning of the first array dimension

end

