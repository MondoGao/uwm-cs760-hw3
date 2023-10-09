# from https://machinelearningmastery.com/cross-entropy-for-machine-learning/#:~:text=Cross%2Dentropy%20is%20a%20measure%20of%20the%20difference%20between%20two,encode%20and%20transmit%20an%20event.
from math import log2
 
def cross_entropy(p, q):
 return -sum([p[i]*log2(q[i]) for i in range(len(p))])
 
p = [0.1, 0.2, 0.7]
q = [0.80, 0.15, 0.05]
# calculate cross entropy H(P, P)
ce_pp = cross_entropy(p, p)
print('H(P, P): %.3f bits' % ce_pp)
# calculate cross entropy H(Q, Q)
ce_qq = cross_entropy(q, q)
print('H(Q, Q): %.3f bits' % ce_qq)
ce_pq = cross_entropy(p, q)
print('H(P, Q): %.3f bits' % ce_pq)