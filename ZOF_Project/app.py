from flask import Flask, render_template, request
from ZOF_CLI import bisection, regula_falsi, secant, newton_raphson, fixed_point_iteration, modified_secant, get_function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    expression = request.form['expression']
    method = request.form['method']
    tol = float(request.form['tol'])
    max_iter = int(request.form['max_iter'])

    f = get_function(expression)
    
    root, error, iterations = None, None, None

    if method == "bisection":
        a = float(request.form['param1'])
        b = float(request.form['param2'])
        root, error, iterations = bisection(f, a, b, tol, max_iter)
    elif method == "regula_falsi":
        a = float(request.form['param1'])
        b = float(request.form['param2'])
        root, error, iterations = regula_falsi(f, a, b, tol, max_iter)
    elif method == "secant":
        x0 = float(request.form['param1'])
        x1 = float(request.form['param2'])
        root, error, iterations = secant(f, x0, x1, tol, max_iter)
    elif method == "newton_raphson":
        x0 = float(request.form['param1'])
        df_str = request.form['df']
        df = get_function(df_str)
        root, error, iterations = newton_raphson(f, df, x0, tol, max_iter)
    elif method == "fixed_point":
        x0 = float(request.form['param1'])
        g_str = request.form['g']
        g = get_function(g_str)
        root, error, iterations = fixed_point_iteration(g, x0, tol, max_iter)
    elif method == "modified_secant":
        x0 = float(request.form['param1'])
        delta = float(request.form['delta'])
        root, error, iterations = modified_secant(f, x0, delta, tol, max_iter)

    return render_template('index.html', root=root, error=error, iterations=iterations, method=method)

if __name__ == '__main__':
    app.run(debug=True)
