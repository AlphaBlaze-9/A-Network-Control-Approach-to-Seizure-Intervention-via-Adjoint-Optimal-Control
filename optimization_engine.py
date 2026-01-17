import cvxpy as cp
import numpy as np
import time

def compute_glasser_control(x, f_x, h_val, grad_h, alpha=0.5):
    # This print statement is for debugging - if you don't see this in your terminal, 
    # it means you are running the wrong file!
    # print("DEBUG: Running Updated Week 5 Solver") 
    
    num_nodes = 360
    u = cp.Variable(num_nodes)
    
    objective = cp.Minimize(0.5 * cp.sum_squares(u))
    drift_term = grad_h @ f_x
    constraints = [drift_term + grad_h @ u >= -alpha * h_val]
    
    prob = cp.Problem(objective, constraints)
    
    start_time = time.time()
    prob.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3)
    solve_time_ms = (time.time() - start_time) * 1000 

    # We are EXPLICITLY returning two values here
    control_vector = u.value if u.value is not None else np.zeros(num_nodes)
    
    return control_vector, solve_time_ms