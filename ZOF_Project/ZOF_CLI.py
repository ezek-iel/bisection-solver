import numpy as np
import argparse

def get_function(expression):
    """
    Create a function from a string expression.
    """
    def f(x):
        return eval(expression, {"x": x, "np": np})
    return f

def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method for finding the root of a function.

    Args:
        f (function): The function for which to find the root.
        a (float): The start of the interval.
        b (float): The end of the interval.
        tol (float): The tolerance for the root.
        max_iter (int): The maximum number of iterations.

    Returns:
        tuple: A tuple containing the root, the error, and the number of iterations.
    """
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None, None, None

    iterations = []
    for i in range(max_iter):
        c = (a + b) / 2
        iterations.append([i + 1, c, f(c), abs(b - a)])
        if abs(f(c)) < tol:
            return c, abs(b - a), iterations
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, abs(b - a), iterations

def regula_falsi(f, a, b, tol=1e-6, max_iter=100):
    """
    Regula Falsi (False Position) method for finding the root of a function.

    Args:
        f (function): The function for which to find the root.
        a (float): The start of the interval.
        b (float): The end of the interval.
        tol (float): The tolerance for the root.
        max_iter (int): The maximum number of iterations.

    Returns:
        tuple: A tuple containing the root, the error, and the number of iterations.
    """
    if f(a) * f(b) >= 0:
        print("Regula Falsi method fails.")
        return None, None, None

    iterations = []
    for i in range(max_iter):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        iterations.append([i + 1, c, f(c), abs(b - a)])
        if abs(f(c)) < tol:
            return c, abs(b - a), iterations
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, abs(b - a), iterations

def secant(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Secant method for finding the root of a function.

    Args:
        f (function): The function for which to find the root.
        x0 (float): The first initial guess.
        x1 (float): The second initial guess.
        tol (float): The tolerance for the root.
        max_iter (int): The maximum number of iterations.

    Returns:
        tuple: A tuple containing the root, the error, and the number of iterations.
    """
    iterations = []
    for i in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        iterations.append([i + 1, x2, f(x2), abs(x2 - x1)])
        if abs(f(x2)) < tol:
            return x2, abs(x2 - x1), iterations
        x0 = x1
        x1 = x2
    return x2, abs(x2 - x1), iterations

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson method for finding the root of a function.

    Args:
        f (function): The function for which to find the root.
        df (function): The derivative of the function.
        x0 (float): The initial guess.
        tol (float): The tolerance for the root.
        max_iter (int): The maximum number of iterations.

    Returns:
        tuple: A tuple containing the root, the error, and the number of iterations.
    """
    iterations = []
    for i in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        iterations.append([i + 1, x1, f(x1), abs(x1 - x0)])
        if abs(f(x1)) < tol:
            return x1, abs(x1 - x0), iterations
        x0 = x1
    return x1, abs(x1 - x0), iterations

def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    """
    Fixed Point Iteration method for finding the root of a function.

    Args:
        g (function): The function x = g(x) for which to find the root.
        x0 (float): The initial guess.
        tol (float): The tolerance for the root.
        max_iter (int): The maximum number of iterations.

    Returns:
        tuple: A tuple containing the root, the error, and the number of iterations.
    """
    iterations = []
    for i in range(max_iter):
        x1 = g(x0)
        iterations.append([i + 1, x1, x1 - g(x1), abs(x1 - x0)])
        if abs(x1 - x0) < tol:
            return x1, abs(x1 - x0), iterations
        x0 = x1
    return x1, abs(x1 - x0), iterations

def modified_secant(f, x0, delta=1e-6, tol=1e-6, max_iter=100):
    """
    Modified Secant method for finding the root of a function.

    Args:
        f (function): The function for which to find the root.
        x0 (float): The initial guess.
        delta (float): A small perturbation value.
        tol (float): The tolerance for the root.
        max_iter (int): The maximum number of iterations.

    Returns:
        tuple: A tuple containing the root, the error, and the number of iterations.
    """
    iterations = []
    for i in range(max_iter):
        x1 = x0 - f(x0) * delta * x0 / (f(x0 + delta * x0) - f(x0))
        iterations.append([i + 1, x1, f(x1), abs(x1 - x0)])
        if abs(f(x1)) < tol:
            return x1, abs(x1 - x0), iterations
        x0 = x1
    return x1, abs(x1 - x0), iterations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero of Functions (ZOF) Solver")
    parser.add_argument("method", help="The root-finding method to use.",
                        choices=["bisection", "regula_falsi", "secant", "newton_raphson", "fixed_point", "modified_secant"])
    parser.add_argument("expression", help="The function to solve, in terms of x.")
    parser.add_argument("params", help="Method-specific parameters.", nargs="+", type=float)
    parser.add_argument("--tol", help="The tolerance for the root.", type=float, default=1e-6)
    parser.add_argument("--max_iter", help="The maximum number of iterations.", type=int, default=100)
    parser.add_argument("--delta", help="Delta for modified secant method.", type=float, default=1e-6)
    parser.add_argument("--df", help="The derivative of the function for Newton-Raphson method.")
    parser.add_argument("--g", help="The g(x) function for Fixed Point Iteration.")

    args = parser.parse_args()

    f = get_function(args.expression)
    
    root, error, iterations = None, None, None

    if args.method == "bisection":
        if len(args.params) != 2:
            print("Bisection method requires two parameters: a and b.")
        else:
            a, b = args.params
            root, error, iterations = bisection(f, a, b, args.tol, args.max_iter)
    elif args.method == "regula_falsi":
        if len(args.params) != 2:
            print("Regula Falsi method requires two parameters: a and b.")
        else:
            a, b = args.params
            root, error, iterations = regula_falsi(f, a, b, args.tol, args.max_iter)
    elif args.method == "secant":
        if len(args.params) != 2:
            print("Secant method requires two parameters: x0 and x1.")
        else:
            x0, x1 = args.params
            root, error, iterations = secant(f, x0, x1, args.tol, args.max_iter)
    elif args.method == "newton_raphson":
        if len(args.params) != 1:
            print("Newton-Raphson method requires one parameter: x0.")
        elif not args.df:
            print("Newton-Raphson method requires the derivative of the function (--df).")
        else:
            df = get_function(args.df)
            x0 = args.params[0]
            root, error, iterations = newton_raphson(f, df, x0, args.tol, args.max_iter)
    elif args.method == "fixed_point":
        if len(args.params) != 1:
            print("Fixed Point Iteration method requires one parameter: x0.")
        elif not args.g:
            print("Fixed Point Iteration method requires the g(x) function (--g).")
        else:
            g = get_function(args.g)
            x0 = args.params[0]
            root, error, iterations = fixed_point_iteration(g, x0, args.tol, args.max_iter)
    elif args.method == "modified_secant":
        if len(args.params) != 1:
            print("Modified Secant method requires one parameter: x0.")
        else:
            x0 = args.params[0]
            root, error, iterations = modified_secant(f, x0, args.delta, args.tol, args.max_iter)

    if root is not None and iterations is not None:
        print(f"\n{args.method.replace('_', ' ').title()} Method:")
        print("Iteration | Root | f(x) | Error")
        for i in iterations:
            print(f"{i[0]:<9} | {i[1]:<20} | {i[2]:<20} | {i[3]:<20}")
        print(f"\nFinal Root: {root}")
        print(f"Final Error: {error}")
        print(f"Iterations: {len(iterations)}")
