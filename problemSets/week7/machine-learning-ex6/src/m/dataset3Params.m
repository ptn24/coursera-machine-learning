function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 1;
% sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vals = C_vals;

opts.cv_error = double(intmax);
opts.C = C_vals(1);
opts.sigma = sigma_vals(1);
for i=1:length(C_vals)
    C_val = C_vals(i);
    for j=1:length(sigma_vals)
        % We first train the SVM classifer on the training set for a
        % particular combination of (`C`, `sigma`).
        sigma_val = sigma_vals(j);
        model = svmTrain(X, y, C_val,...
                         @(x1, x2) gaussianKernel(x1, x2, sigma_val));

        % We next use the trained model to classify the cross validation
        % set.
        h = svmPredict(model, Xval);

        % We next compute the cross validation error and update the optimal
        % param values.
        cv_error = mean(double(h ~= yval));
        if cv_error <= opts.cv_error
            opts.cv_error = cv_error;
            opts.C = C_val;
            opts.sigma = sigma_val;
        end
    end
end

C = opts.C;
sigma = opts.sigma;

% =========================================================================

end
