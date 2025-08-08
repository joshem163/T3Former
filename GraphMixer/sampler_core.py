import numpy as np

class MockRet:
    def row(self):
        return np.array([0, 1], dtype=np.int32)

    def col(self):
        return np.array([1, 2], dtype=np.int32)

    def eid(self):
        return np.array([10, 11], dtype=np.int32)

    def nodes(self):
        return np.array([0, 1, 2], dtype=np.int32)

    def ts(self):
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)


class ParallelSampler:
    def __init__(self, *args, **kwargs):
        print("⚠️ Warning: Using mock ParallelSampler (no real sampling).")

    def sample(self, nodes, ts):
        pass

    def get_ret(self):
        return [MockRet()]
    def reset(self):
        # Add this to fix the AttributeError
        pass
