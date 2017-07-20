function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
%fprintf(['size(p) => %d\n'], size(p));
% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
%fprintf(['size of all_theta*XT => %d\n'], size(all_theta*X'));
%all_theta
%X(4000:4500, 1:20)
    %for i=1:size(p, 1)
        %fprintf(['i = %d\n'], i);
        %fprintf(['size of all_theta(i, :) => %d\n'], size(all_theta(i,
        %:)));
        %all_theta
        %h = sigmoid(all_theta * X');
        h = sigmoid((all_theta*X')');
        [val_max, idx_max] = max(h, [], 2);
        %fprintf(['size i_max = %d\n'], size(idx_max));
        p = idx_max;
        
        %(all_theta * X')'
        %fprintf(['size of val = %d\n'], size(val));
        %fprintf(['size of idx = %d\n'], size(idx));
        
        %if (i < 10)
           %fprintf(['i = %d, idx = %d and val = %d\n'], i, idx, val);
        %end
    %end
  %p(i, 1) = all_theta(i, idx);



% =========================================================================


end
