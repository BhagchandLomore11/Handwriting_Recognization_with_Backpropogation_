function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m, 1), X]; % adding bias unit to X % map from layer 1 to layer 2

z2 = a1 * Theta1'; % Convert into 5000 X 26

a2 = sigmoid(z2); % Convert into 0 to 1 value

a2 = [ones(m, 1), a2]; % Add ones to the h1 data matrix

z3 = a2 * Theta2'; % Converts to matrix of 5000 exampls x num_labels

a3 = sigmoid(z3); % Convert into 0 to 1 value

h = a3;

Y = zeros(m,num_labels);

dis = zeros(m,1);

for i = 1:m
    Y(i,y(i)) = 1;
    dis(i) = log(h(i,:))*(-Y(i,:))' - log(1-h(i,:))*(1-Y(i,:))';
end

J = 1/m * sum(dis);

%add the cost for the regularization terms

J = J + lambda/(2*m)*( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );

%Backpropagation
for t = 1:m
    a1 = [1; X(t,:)'];
 
    z2 = Theta1 * a1;
    a2 = [1;sigmoid(z2)];
 
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
 
    delta3 = a3 - Y(t,:)';
 
    delta2 = (Theta2'*delta3).*[1;sigmoidGradient(z2)];
    delta2 = delta2(2:end);
 
    Theta1_grad = Theta1_grad + delta2*a1';
    Theta2_grad = Theta2_grad + delta3*a2';
end

Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;

%regularization
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
