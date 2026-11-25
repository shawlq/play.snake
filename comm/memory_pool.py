import random
from collections import deque


class Memory2D:
    """二维经验表：维度1=L(蛇长)，维度2=HC是否存在(0/1)，内部为deque maxlen=10000"""
    def __init__(self, maxlen=10000):
        self.maxlen = maxlen
        self._deques = {}  # (L, hc) -> deque

    def __key(self, L, hc):
        return (L, hc)

    def _get_deque(self, k):
        if k not in self._deques:
            dq = deque(maxlen=self.maxlen)
            self._deques[k] = dq
        return self._deques[k]

    def append(self, item, L, hc):
        """item: (state, action, reward, next_state, done)"""
        self._get_deque(self.__key(L, hc)).append(item)

    def __len__(self):
        return sum(len(dq) for dq in self._deques.values())

    def sample(self, n):
        all_items = []
        for dq in self._deques.values():
            if len(dq) > 0:
                all_items.extend(random.sample(dq, min(len(dq), n)))
        return all_items

    def get_counts(self):
        """返回各 (L, hc) 对应的数据量，格式 {(L, hc): num}"""
        return {k: len(dq) for k, dq in self._deques.items()}
