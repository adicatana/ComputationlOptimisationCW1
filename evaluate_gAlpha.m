function gA = evaluate_gAlpha(beta_vect, X, y, n, m, dim, lambda, eval, norm, step_size)

gr = evaluate_gB(beta_vect, X, y, n, m, dim, lambda, eval, norm);
value_with_alpha = beta_vect - step_size * gr;

grad_2 = -evaluate_gB(value_with_alpha, X, y, n, m, dim, lambda, eval, norm);

gA = dot(gr, grad_2);
