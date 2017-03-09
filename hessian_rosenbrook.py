'''
https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/optimize.html
Hessian product example:
For larger minimization problems, storing the entire Hessian matrix can consume considerable time and memory. The Newton-CG algorithm only needs the product of the Hessian times an arbitrary vector. As a result, the user can supply code to compute this product rather than the full Hessian by giving a hess function which take the minimization vector as the first argument and the arbitrary vector as the second argument (along with extra arguments passed to the function to be minimized). If possible, using Newton-CG with the Hessian product option is probably the fastest way to minimize the function.

In this case, the product of the Rosenbrock Hessian with an arbitrary vector is not difficult to compute. If pp is the arbitrary vector, then H(x)pH(x)p has elements:

H(x)p=⎡⎣⎢⎢⎢⎢⎢⎢(1200x20−400x1+2)p0−400x0p1⋮−400xi−1pi−1+(202+1200x2i−400xi+1)pi−400xipi+1⋮−400xN−2pN−2+200pN−1⎤⎦⎥⎥⎥⎥⎥⎥.
H(x)p=[(1200x02−400x1+2)p0−400x0p1⋮−400xi−1pi−1+(202+1200xi2−400xi+1)pi−400xipi+1⋮−400xN−2pN−2+200pN−1].
Code which makes use of this Hessian product to minimize the Rosenbrock function using minimize follows:
'''

def rosen_hess_p(x, p):
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
    Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
               -400*x[1:-1]*p[2:]
    Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
    return Hp

res = minimize(rosen, x0, method='Newton-CG',
               jac=rosen_der, hessp=rosen_hess_p,
               options={'xtol': 1e-8, 'disp': True})






res.x