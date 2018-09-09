function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % ============================================================


	% calculating the sum
	relativeTheta1 = 0;
	relativeTheta2 = 0;
	for i = 1 : m
		relativeTheta1 = relativeTheta1 + (1 / m) * (hypothesis(X(i, :)', theta) - y(i));
		relativeTheta2 = relativeTheta2 + (1 / m) * (hypothesis(X(i, :)', theta) - y(i)) * X(i, 2);
	end

        tempTheta1 = theta(1) - (alpha * relativeTheta1);
        tempTheta2 = theta(2) - (alpha * relativeTheta2);

        theta(1) = tempTheta1;
        theta(2) = tempTheta2;
    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end

function h = hypothesis(X, theta)
  
  h =  theta' * X;
end