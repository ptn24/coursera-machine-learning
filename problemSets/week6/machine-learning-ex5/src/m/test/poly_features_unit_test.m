function [ output_args ] = poly_features_unit_test( input_args )
%POLY_FEATURES_UNIT_TEST Summary of this function goes here
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

function test_power_one(testCase)
x = [1; 2; 3];
p = 1;
verifyEqual(testCase, polyFeatures(x, p), x);
end

function test_power_two(testCase)
x = [1; 2; 3];
p = 2;
exp_X_poly = [1 1;
              2 4;
              3 9];
verifyEqual(testCase, polyFeatures(x, p), exp_X_poly);
end
