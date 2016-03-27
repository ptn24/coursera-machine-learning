function [ output_args ] = sigmoid_gradient_unit_test( input_args )
%SIGMOID_GRADIENT_UNIT_TEST Summary of this function goes here
%   Detailed explanation goes here
output_args = functiontests(localfunctions);
end

function setup(testCase)
testCase.TestData.originalDir = pwd;
cd('../');
end

function teardown(testCase)
cd(testCase.TestData.originalDir);
end

function testBasic(testCase)
x = 8;
expSigmoidGradient = sigmoid(x) * (1 - sigmoid(x));
verifyEqual(testCase, sigmoidGradient(x), expSigmoidGradient);
end