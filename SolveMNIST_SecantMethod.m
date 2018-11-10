function ReturnVal = SolveMNIST_SecantMethod(tol, num_iter, step_size, lambda)

% Initialise the training set --------------------------------------------
load mnist.mat
n   = 1000; % Input features
m   = 1000; % Test cases
dim =   10;

% l-2 Regulariser
norm_type = 2;

% Initialise a starting point for the algorithm --------------------------
current_variable_value = zeros(1,n*dim);

current_func_value = evaluate_gB(current_variable_value, X, y, n, m, dim, lambda, 0, norm_type);
current_variable_gradient = evaluate_gB(current_variable_value, X, y, n, m, dim, lambda, 1, norm_type);

% Store beta guesses at each iteration
variable_values_vector(1,:) = current_variable_value;

% Store the function value at each iteration
function_evaluation_vector(1) = current_func_value;

previous_step_size = step_size;

% first_deriv = evaluate_gAlpha(current_variable_value, X, y, n, m, dim, lambda, 1, norm_type, step_size);
% second_deriv = evaluate_gAlpha(first_deriv, X, y, n, m, dim, lambda, 1, norm_type, step_size);
%
% second_step_size = step_size - first_deriv / second_deriv;
second_step_size = 0.00005;

current_step_size = second_step_size;

curr_step_size = 0.0001;

fprintf('\niter=%d; Func Val=%f; FONC Residual=%f', 0, current_func_value, norm(current_variable_gradient));

% Iterative algorithm begins ---------------------------------------------
for i = 1:num_iter

    previous_variable_gradient = current_variable_gradient;

    % Step for gradient descent ------------------------------------------
    current_variable_value = variable_values_vector(i,:) - curr_step_size * previous_variable_gradient;

    % Update with the new iteration --------------------------------------
    variable_values_vector(i+1,:) = current_variable_value;

    current_func_value = evaluate_gB(current_variable_value, X, y, n, m, dim, lambda, 0, norm_type);

    function_evaluation_vector(i+1) = current_func_value;

    current_variable_gradient = evaluate_gB(current_variable_value, X, y, n, m, dim, lambda, 1, norm_type);

    % Check if it's time to terminate ------------------------------------

    % Check the FONC?
    % Store the norm of the gradient at each iteration
    convgsd(i) = norm(previous_variable_gradient, 2);

    % Check that the vector is changing from iteration to iteration?
    % Stores length of the difference between the current beta and the
    % previous one at each iteration
    lenXsd(i)  = norm(current_variable_value - variable_values_vector(i,:), 2);

    % Check that the objective is changing from iteration to iteration?
    % Stores the absolute value of the difference between the current
    % function value and the previous one at each iteration
    diffFsd(i) = abs(current_func_value - function_evaluation_vector(i));

    fprintf('\niter=%d; Func Val=%f; FONC Residual=%f; Sqr Diff=%f',...
            i, current_func_value, convgsd(i), lenXsd(i));

    curr_step_size = getNextStepSize(X, y, n, m, dim, lambda, 1, norm_type, 0.0001, 0.00005, current_variable_value);
    %previous_step_size = current_step_size;
    %current_step_size = next_step_size;

    % Check the convergence criteria?
    if (convgsd(i) <= tol)
        fprintf('\nFirst-Order Optimality Condition met\n');
        break;
    elseif (lenXsd(i) <= tol)
        fprintf('\nExit: Design not changing\n');
        break;
    elseif (diffFsd(i) <= tol)
        fprintf('\nExit: Objective not changing\n');
        break;
    elseif (i + 1 >= num_iter)
        fprintf('\nExit: Done iterating\n');
        break;
    end

end

ReturnVal = current_variable_value;

l = length(function_evaluation_vector);

plt = plot(1:l, function_evaluation_vector);
saveas(plt, "plot_secant_5000.jpg");

end
