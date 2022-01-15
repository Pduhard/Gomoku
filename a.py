from re import A
from time import perf_counter


def itera(a):
	for i in range(900):
		a += a
	return a

def reca(a, lp):
	return (a if lp == 900 else a + reca(a, lp + 1))


t = perf_counter()
itera(1)
print(perf_counter() - t)
t = perf_counter()
reca(1, 0)
print(perf_counter() - t)