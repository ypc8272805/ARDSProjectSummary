import hyperopt as hp
from hyperopt import hp, fmin, rand, tpe, space_eval


# 定义目标函数
def q(args):
    x, y = args
    return x ** 2 + y ** 2


# 定义配置空间
space = [hp.uniform('x', -1, 1), hp.normal('y', -1, 1)]
# 选择一个搜索算法
best = fmin(q, space, algo=tpe.suggest, max_evals=100)
print(best)
print(space_eval(space, best))

import pickle
import time
from hyperopt import STATUS_OK


def objective(x):
    return {'loss': x ** 2, 'status': STATUS_OK}


best = fmin(objective,
            space=hp.uniform('x', -10, 10),
            algo=tpe.suggest,
            max_evals=100)
print(best)

### The Trials Object
import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def objective(x):
    return {
        'loss': x ** 2,
        'status': STATUS_OK,
        'eval_time': time.time(),
        'other_stuff': {'type': None, 'value': [0, 1, 2]},
        'attachments':
            {'time_module': pickle.dumps(time.time)}
    }


trials = Trials()
best = fmin(objective,
            space=hp.uniform('x', -10, 10),
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
print(best)

space = hp.choice('a',
                  [
                      ('case 1', 1 + hp.lognormal('c1', 0, 1)),
                      ('case 2', hp.uniform('c2', -10, 10))
                  ])

import hyperopt.pyll.stochastic

print(hyperopt.pyll.stochastic.sample(space))
