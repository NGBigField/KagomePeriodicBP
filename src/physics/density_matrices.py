import numpy as np

#TODO Check
def random_quantum_state_tensor(dims:tuple[int,...]) -> np.ndarray:
    rand = lambda: np.random.rand(*dims)
    m = rand() + 1j*rand()
    # m_h = m.transpose()
    # m_h = m_h.conjugate()
    m_h = m.conjugate()
    m *= m_h
    m /= m.trace()
    return m

def random_valid_density_matrix(n:int) -> np.ndarray:
    rand = lambda n_: np.random.rand(n_, n_)
    m = rand(n) + 1j*rand(n)
    m_h = m.transpose()
    m_h = m_h.conjugate()
    m *= m_h
    m /= m.trace()
    return m


def _test():
    n = 2
    m = random_valid_density_matrix(n)
    print(m)


if __name__ == "__main__":
    _test()