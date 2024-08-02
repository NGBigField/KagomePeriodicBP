import numpy as np

def _random_quantum_state_tensor(dims:tuple[int,...]) -> np.ndarray:
    _rand = lambda: np.random.rand(*dims)
    m = _rand() + 1j*_rand()
    # m_h = m.transpose()
    # m_h = m_h.conjugate()
    m_h = m.conjugate()
    m *= m_h
    m /= m.trace()
    return m