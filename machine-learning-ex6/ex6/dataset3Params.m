function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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
x1 = [1 2 1]; x2 = [0 4 -1]; 
test_values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
C = test_values(1);
sigma = test_values(1);

% the first prediction error for reference to compare
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
predictions = svmPredict(model, Xval);
minPredictionError = mean(double(predictions ~= yval));

number_of_test_values = size(test_values, 1);
for indexC = 1 : number_of_test_values
	loopC = test_values(indexC);
	for indexSigma = 1 : number_of_test_values
		loopSigma = test_values(indexSigma);
		fprintf('training with loopC = %f and loopSigma = %f\n', loopC, loopSigma);
		model= svmTrain(X, y, loopC, @(x1, x2) gaussianKernel(x1, x2, loopSigma)); 
		predictions = svmPredict(model, Xval);
		loopPredictionError = mean(double(predictions ~= yval));
		if (loopPredictionError < minPredictionError)
			fprintf('new minimal error: %f\n', loopPredictionError);
			minPredictionError = loopPredictionError;
			C = loopC;
			sigma = loopSigma;
			fprintf('sigma = %f\nC = %f\n', sigma, C);
		end
	end
end

fprintf('Final sigma = %f\nFinal C = %f\n', sigma, C);

[C, sigma];







% =========================================================================

end
