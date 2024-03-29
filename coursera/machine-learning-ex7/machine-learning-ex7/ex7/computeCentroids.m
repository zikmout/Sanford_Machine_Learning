function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%Randomly initialize K cluster centroids mu(1), mu(2), ..., mu(K)
%Repeat:
%    for i = 1 to m:
%       c(i):= index (from 1 to K) of cluster centroid closest to x(i)
%    for k = 1 to K:
%       mu(k):= average (mean) of points assigned to cluster k

counter = zeros(K,1);

for i = 1:m
    for k = 1:K
        if idx(i) == k
            centroids(k,:) = centroids(k,:) + X(i,:);
            counter(k) = counter(k) + 1;
        end
    end
end

for k = 1:K
    centroids(k,:) = centroids(k,:) ./ counter(k);
end






% =============================================================


end

