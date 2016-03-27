function [ o ] = nnCostFunctionUtils( ~ )
%NNCOSTFUNCTIONUTILS Summary of this function goes here
%   Detailed explanation goes here

o.labelVectorToMatrix = @labelVectorToMatrix;
o.forwardPropagateOne = @forwardPropagateOne;
o.backPropagateOne = @backPropagateOne;

end

function [ label_matrix ] = labelVectorToMatrix(label_vector, num_labels)
%Converts a label vector of scalars into a label matrix with logical row
%vectors.
%   That is, this function converts
%
%     `label_vector` = [1; 2; 3; 4; 5; 6; 7; 8; 9; 0]
%     `num_labels` = 10
%
%   to
%
%     `label_matrix` = [
%       [1 0 0 0 0 0 0 0 0 0];
%       [0 1 0 0 0 0 0 0 0 0];
%       [0 0 1 0 0 0 0 0 0 0];
%       [0 0 0 1 0 0 0 0 0 0];
%       [0 0 0 0 1 0 0 0 0 0];
%       [0 0 0 0 0 1 0 0 0 0];
%       [0 0 0 0 0 0 1 0 0 0];
%       [0 0 0 0 0 0 0 1 0 0];
%       [0 0 0 0 0 0 0 0 1 0];
%       [0 0 0 0 0 0 0 0 0 1]   Note that label 0 (int) maps to row vector
%                               with a 1 (int) in elt with index 10.
%     ]

m = length(label_vector);
label_matrix = zeros(m, num_labels);
for i=1:m
    label_vector_ = zeros(num_labels, 1);
    label_vector_(label_vector(i)) = 1;
    label_matrix(i, :) = label_vector_';
end
end

function [ A_1, Z_1 ] = forwardPropagateOne( Theta_0, A_0 )
%Computes the results of propagating `A_0`, weighted by `Theta_0`, a single
%level in a neural network.
%
%   A_0: An (m x n) matrix, where m is the number of samples and n is the
%   number of features per sample.
%
%   Theta_0: An (o x (n + 1)) matrix where (n + 1) is the number of inputs
%   (including the additional biased input) and o is the number of outputs.
%   `Theta_0[j][2:n+1]` specifies the weights of `X[i][:]` in computing
%   `Y[j][:]`.

m = size(A_0, 1);
A_0 = [ ones(m, 1) A_0 ];  % Append the biased input for each sample.
Z_1 = Theta_0 * A_0';  % We tranpose `A_0` here, so each sample is now a
                       % column.
A_1 = sigmoid(Z_1);
A_1 = A_1';  % We transpose here again to represent each sample as a row.
Z_1 = Z_1';
end

function [ Delta_0_ ] = backPropagateOne( Theta_0, Delta_1_, Z_0 )
Delta_0_ = Theta_0(:, 2:end)' * Delta_1_' .* sigmoidGradient(Z_0');
Delta_0_ = Delta_0_';
end