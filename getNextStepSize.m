function step_size = getNextStepSize(X, y, n, m, dim, lambda, eval, norm, prev_step, curr_step, curr_value)

iterations = 10;

for i = 1:iterations
  if (abs(prev_step - curr_step) < 0.00001)
    break
  end


  current_gradient   = evaluate_gAlpha(curr_value, X, y, n, m, dim, lambda, 1, norm, curr_step);
  previous_gradient  = evaluate_gAlpha(curr_value, X, y, n, m, dim, lambda, 1, norm, prev_step);

  if abs(current_gradient-previous_gradient) < 0.000001
    break
  end

  aux_step_size      = curr_step - current_gradient * ((curr_step - prev_step)/(current_gradient - previous_gradient));
  prev_step = curr_step;
  curr_step  = aux_step_size;
end

step_size = curr_step;
