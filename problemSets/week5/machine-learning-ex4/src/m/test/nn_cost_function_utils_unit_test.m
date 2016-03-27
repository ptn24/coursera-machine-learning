function [ output_args ] = nn_cost_function_utils_unit_test( ~ )
%NN_COST_FUNCTION_UNIT_TEST Summary of this function goes here
%   Detailed explanation goes here
output_args = functiontests(localfunctions);
end

function setup(testCase)
testCase.TestData.originalDir = pwd;
cd('../');

% Import util funcs.
testCase.TestData.nnCostFunctionUtils = nnCostFunctionUtils();

testCase.TestData.foo = 2;
end

function teardown(testCase)
cd(testCase.TestData.originalDir);
end

function testSetup(testCase)
verifyEqual(testCase, testCase.TestData.foo, 2);
end

function testLabelVectorToMatrix(testCase)
labelVectorToMatrix = testCase.TestData.nnCostFunctionUtils...
    .labelVectorToMatrix;

labelVector = [1; 2; 3; 4];
numLabels = 4;
expLabelMatrix = [
    [1 0 0 0];
    [0 1 0 0];
    [0 0 1 0];
    [0 0 0 1]
];
verifyEqual(testCase,...
            labelVectorToMatrix(labelVector, numLabels),...
            expLabelMatrix)
end

function testForwardPropagateOne(testCase)
forwardPropagateOne = testCase.TestData.nnCostFunctionUtils...
    .forwardPropagateOne;

Theta = [
    1 2 3;
    4 5 6
];

A_1 = [
    1 2;
    3 4
];

% The computations here account for the bias units.
exp_Z_2 = [
    (1 * Theta(1, 1) + A_1(1, 1) * Theta(1, 2) +...
        A_1(1, 2) * Theta(1, 3)),...
    (1 * Theta(2, 1) + A_1(1, 1) * Theta(2, 2) + A_1(1, 2) * Theta(2, 3));

    (1 * Theta(1, 1) + A_1(2, 1) * Theta(1, 2) +...
        A_1(2, 2) * Theta(1, 3)),...
    (1 * Theta(2, 1) + A_1(2, 1) * Theta(2, 2) + A_1(2, 2) * Theta(2, 3))
];
exp_A_2 = sigmoid(exp_Z_2);

[A_2, Z_2] = forwardPropagateOne(Theta, A_1);
verifyEqual(testCase, Z_2, exp_Z_2);
verifyEqual(testCase, A_2, exp_A_2);
end

function testBackwardPropagateOne(testCase)
f = testCase.TestData.nnCostFunctionUtils.backPropagateOne;
Theta = [
    1 2 3;
    4 5 6
];
Z_0 = [
    1 2;
    3 4
];
Delta_1_ = [
    2 4;
    8 16
];

exp_Delta_0_ = [
    ((Delta_1_(1, 1) * Theta(1, 2) + Delta_1_(1, 2) * Theta(2, 2)) *...
     sigmoidGradient(Z_0(1, 1))),...
    ((Delta_1_(1, 1) * Theta(1, 3) + Delta_1_(1, 2) * Theta(2, 3)) *...
     sigmoidGradient(Z_0(1, 2)));

    ((Delta_1_(2, 1) * Theta(1, 2) + Delta_1_(2, 2) * Theta(2, 2)) *...
     sigmoidGradient(Z_0(2, 1))),...
    ((Delta_1_(2, 1) * Theta(1, 3) + Delta_1_(2, 2) * Theta(2, 3)) *...
     sigmoidGradient(Z_0(2, 2)))
];

Delta_0_ = f(Theta, Delta_1_, Z_0);
verifyEqual(testCase, Delta_0_, exp_Delta_0_);
end