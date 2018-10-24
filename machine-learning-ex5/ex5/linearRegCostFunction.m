function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%J = ((1/2*m) * sum(((X * theta) - y).^2)) + ((lambda/2*m) * ((sum(theta.^2) - (theta(1)^2))));

for i = 1:m
	J = J + (X(i, :) * theta - y(i)) ^ 2;
end

J = J/(2 * m);

regularization = 0;

for j = 2:size(theta)
	regularization = regularization + theta(j) ^ 2;
end

regularization = regularization * (lambda/(2 * m));

J = J + regularization;

for i = 1:m
	grad(1) = grad(1) + (X(i, :) * theta - y(i)) * X(1, 1);
end
	grad(1) = grad(1) / m;
for j = 2:size(grad)
	for i = 1:m
		grad(j) = grad(j) + (X(i, :) * theta - y(i)) * X(i, j);
	end
	grad(j) = grad(j) / m;
	grad(j) = grad(j) + (lambda/m) * theta(j);
end
% =========================================================================

grad = grad(:);

end
