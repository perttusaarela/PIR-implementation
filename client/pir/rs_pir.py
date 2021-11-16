from typing import List
import asyncio

from sage.all import vector, matrix, codes, VectorSpace, LCM, FiniteField


from sage.misc.persist import dumps, loads
import aiohttp

from utils import evaluate, polyencode, get_urls

setup_complete = False
store_complete = False

F = FiniteField(256)


def setup(
    num_servers: int,
    file_len: int,
    num_colluding: int,
    num_byzantine: int,
    num_unresponsive: int,
) -> None:
    """Define global variables based on the parameters.
    
    """

    global L, S, G, C, D, alpha, rho, n, k, t, b, r, CorrectingCode, URLS

    n = num_servers
    k = file_len
    t = num_colluding
    b = num_byzantine
    r = num_unresponsive

    rho = n - (k + t + 2 * b + r - 1)
    # S = ceil(L*k / rho)
    S = LCM(rho, k) / rho
    L = LCM(rho, k) / k
    alpha = F.list()[:n]
    alpha = vector(F, alpha)
    beta_C = vector(F, [1] * n)
    beta_D = vector(F, [1] * n)

    URLS = get_urls(n, t)

    # Codes:
    C = codes.GeneralizedReedSolomonCode(alpha, k, beta_C)
    D = codes.GeneralizedReedSolomonCode(alpha, t, beta_D)
    G = C.generator_matrix()
    CorrectingCode = codes.GeneralizedReedSolomonCode(alpha, n - 2 * b - r, beta_C)

    global setup_complete
    setup_complete = True


async def store_server(session, url, dat):

    data = aiohttp.FormData()

    data.add_field("query", dumps(dat))

    async with session.post(url, data=data) as res:

        if res.status != 200:
            raise RuntimeError("network issue")

    return True


async def store_async(data: List[bytes]) -> None:
    """Store the given data on the servers
    
    """

    assert setup_complete, "setup must be complete before store"

    global num_files

    num_files = len(data)

    X = matrix(F, L * num_files, k, 0)
    V = VectorSpace(F, k)
    for i in range(0, L * num_files):
        X[i] = V.random_element()
    Y = X * G

    async with aiohttp.ClientSession() as session:

        tasks = []

        for url, data in zip(URLS, (Y[i] for i in range(n))):
            tasks.append(asyncio.ensure_future(store_server(session, url, data)))

        for res in asyncio.as_completed(tasks):
            await res

    global store_complete
    store_complete = True


def store(data: List[bytes]) -> None:

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(store_async(data))


async def query_server(session, url, query, sid):

    data = aiohttp.FormData()

    data.add_field("query", dumps(query))

    async with session.post(url, data=data) as res:

        if res.status != 200:
            raise RuntimeError("network issue")

        r = loads(await res.text())

    return sid, r


async def retrieve_async(index: int) -> bytes:

    # Queries:
    Q = matrix(F, L * num_files, n, 0)

    # Iterating over S rounds:

    R = F["x"]
    x = R.gen()

    h = matrix(F, S, rho, 0)

    async with aiohttp.ClientSession() as session:

        for s in range(S):

            block = s * L
            subs = vector(F, [0] * n)

            # random queries for this round
            for i in range(0, L * num_files):
                Q[i] = D.random_element()

            # generating the retrieval vector
            e = matrix(F, L, n, 0)
            for l in range(L):
                p = (s + 1) * rho + (-l) * k + t - 1
                if p >= 0:
                    e[l] = evaluate(x ** p, alpha)

            # polynomial for removing already known information
            if s > 0:
                for sigma in range(s):
                    exponent = rho * (s - sigma) + k + t - 1
                    subs = subs + evaluate(
                        (x ** exponent) * polyencode(R, vector(F, h[sigma].list())),
                        alpha,
                    )

            # Querying the servers
            E = matrix(F, num_files * L, n, 0)
            if L == 1:
                E[index] = e
            else:
                for l in range(L):
                    E[index * L + l] = e[l]
            queries = Q + E
            response = vector(F, [0] * n)

            tasks = []

            for url, query, sid in zip(URLS, (queries[i] for i in range(n)), range(n)):
                tasks.append(
                    asyncio.ensure_future(query_server(session, url, query, sid))
                )

            for res in asyncio.as_completed(tasks):
                i, r = await res
                response[i] = r

            ret0 = response - subs

            # Error-correction:
            ret1 = CorrectingCode.decode_to_code(ret0)

            # Decoding:
            points = list(zip(alpha, ret1))
            poly = R.lagrange_polynomial(points)
            coeffs = poly.list()[k + t - 1 : n - 2 * b - r + 1]
            if len(coeffs) < rho:
                zero_list = [0] * (rho - len(coeffs))
                coeffs = coeffs + zero_list
            h[s] = vector(F, coeffs)

    ans = matrix(F, 1, L * k, 0)
    for i in range(S):
        for j in range(rho):
            ans[0, i * rho + j] = h[i, rho - j - 1]
    answer = matrix(F, L, k, ans.list())
    for i in range(L):
        answer[i] = answer[i][::-1]
    lst = list(range(index * L, index * L + L))

    return


def retrieve(index: int) -> bytes:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(retrieve_async(index))
