function [J, grad] = nnCostFunction(nn_params,...
                                    input_layer_size,...
                                    hidden_layer_size,...
                                    num_labels,...
                                    X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight
% matrices for our 2 layer neural network
Theta1 = reshape(...
    nn_params(1:hidden_layer_size * (input_layer_size + 1)),...
    hidden_layer_size,...
    (input_layer_size + 1)...
);

Theta2 = reshape(...
    nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),...
    num_labels,...
    (hidden_layer_size + 1)...
);

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
% J = 0;
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial
%         derivatives of the cost function with respect to Theta1 and
%         Theta2 in Theta1_grad and Theta2_grad, respectively. After
%         implementing Part 2, you can check that your implementation is
%         correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector
%               into a binary vector of 1's and 0's to be used with the
%               neural network cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for
%               the first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to
%               Theta1_grad and Theta2_grad from Part 2.
%

utils = nnCostFunctionUtils();

[A_2, Z_2] = utils.forwardPropagateOne(Theta1, X);

% `A_3` is matrix where each row is a sample and row[j] is the likelihood
% that that sample belongs to class j.
[A_3, ~] = utils.forwardPropagateOne(Theta2, A_2);

Y = utils.labelVectorToMatrix(y, num_labels);

% Unroll `A3` and `Y` for easy summation over all samples.
a3_vectorized = A_3(:);
y_vectorized = Y(:);
J = (-log(a3_vectorized') * y_vectorized) +...
      (-log(1 - a3_vectorized') * (1 - y_vectorized));
J = J / m;  % `J` without regularization.

% Unroll all "Thetas", omitting the biased weights, for easy summation.
Theta1_unbiased = Theta1(:, 2:end);
Theta2_unbiased = Theta2(:, 2:end);
theta_vectorized = [ Theta1_unbiased(:); Theta2_unbiased(:) ];

% Regularize `J`.
J = J + (lambda / (2 * m)) * (theta_vectorized' * theta_vectorized);

% =========================================================================

Delta_3_ = A_3 - Y;

% `Delta_2_` is matrix where each row represents the "errors" that a
% particular sample has in layer 2 of the neural network and row[j] is the
% "error" associated with node j in layer 2 for that sample.
%
% This is the crux of the backpropagation algo. Note that we must first
% remove the bias units from `Theta2` here, or in the helper function that
% performs the backpropagation algo.
Delta_2_ = utils.backPropagateOne(Theta2, Delta_3_, Z_2);

% There is no `Delta_1_` because we do not associate "errors" with inputs.

% We must append the bias units for backpropagation.
one_and_A_2 = [ones(size(A_2, 1), 1), A_2];
one_and_A_1 = [ones(size(X, 1), 1), X];

Delta_2 = 0;
Delta_1 = 0;
for t=1:m
    col_Delta_3_ = Delta_3_(t, :)';  % (i.e., transpose row of `Delta_3_`)
    delta_2_ = Delta_2_(t, :)';

    row_A_2 = one_and_A_2(t, :);
    row_A_1 = one_and_A_1(t, :);

    % Accumulate `Delta_*` for each sample.
    % TODO: Can we vectorize this?
    Delta_2 = Delta_2 + col_Delta_3_ * row_A_2;
    Delta_1 = Delta_1 + delta_2_ * row_A_1;
end

% Note that `Theta*_grad` must have the same dimensions as `Theta*`.
Theta2_grad = Delta_2 / m;
Theta1_grad = Delta_1 / m;

% Regularize.
c = lambda / m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + c * Theta2(:, 2:end);
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + c * Theta1(:, 2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end