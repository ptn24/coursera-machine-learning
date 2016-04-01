function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;
% grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%% Compute `J`.
% `X` already has the column-vector of ones prepended.
H = X * theta;

k = 1 / 2 / m;
cost_term = k * (H - y)' * (H - y);

k = k * lambda;
regularization_term = k * theta(2:end)' * theta(2:end);

J = cost_term + regularization_term;

%% Compute `grad`.
k = 1 / m;
unregularized_grad = k * X' * (H - y);

k = lambda / m;
grad_regularization_term = k * theta;
grad_regularization_term(1) = 0;  % Do not regularize "theta_0".
grad = unregularized_grad + grad_regularization_term;

% =========================================================================

grad = grad(:);

end
