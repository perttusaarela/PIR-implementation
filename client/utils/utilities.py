# Generally usefull functions

from sage.all import matrix, vector

# Returns an invertible submatrix of A and the correspoding
# incidence vector.


def invertible_submatrix(A):
    lng = A.ncols()
    shrt = A.nrows()
    ret_mat = matrix(F, shrt, shrt, 0)
    ret_vec = vector(F, [0] * lng)
    for i in range(shrt):
        for j in range(lng):
            if A[i, j] != 0:
                ret_mat[:, i] = A[:, j]
                ret_vec[j] = 1
                break
    if ret_mat.rank() == shrt:
        return ret_mat, ret_vec
    else:
        print("No invertible submatrix found")
        return ret_mat, ret_vec


# Star product of two vectors:


def star(a, b):
    return a.pairwise_product(b)


# evaluation vector eval, such that eval_i = f(a_i)


def evaluate(f, a):
    F = a.base_field()
    ret = vector(F, [f(ai) for ai in a])
    return ret


def polyencode(R, points):
    x = R.gen()
    ret = x ** 0 - 1
    for i in range(points.length()):
        ret += points[i] * x ** i
    return ret
