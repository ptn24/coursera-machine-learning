function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

% Compute the matrix of predictions for each (movie, user) combination.
% `H(i, j)` represents the predicted rating of movie_i by user_j.
H = X * Theta';

% Filter to only compare the predicted and actual ratings for movies that
% were actually rated in the dataset (i.e., known; the (movie_i, user_j)
% combinations st. `R(i, j) == 1`).
H_known = H(R == 1);
Y_known = Y(R == 1);

% Compute the unregularized collaborative filtering cost function.
J = (H_known - Y_known)' * (H_known - Y_known) / 2;

% Add the regularization term.
r = (lambda / 2) * (sum(sum(Theta .^ 2)) + sum(sum(X .^ 2)));
J = J + r;

% Compute the unregularized gradients.
X_grad = ((H .* R) - (Y .* R)) * Theta;
Theta_grad = ((H .* R) - (Y .* R))' * X;

% Add the regularization terms.
X_r = lambda * X;
X_grad = X_grad + X_r;
Theta_r = lambda * Theta;
Theta_grad = Theta_grad + Theta_r;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
