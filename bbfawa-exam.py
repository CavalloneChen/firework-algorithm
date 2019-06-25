import fwa.BBFWA as BBFWA
import benchmarks.cec2017.cec17 as cec17
# import benchmarks.cec2013.cec13 as cec13
import sys
import numpy as np
import pickle as pkl
from tqdm import tqdm
import ray

sys.path.append("../")

if __name__ == "__main__":

    exp_info = {
        "algorihtm": "BBFWA",
        "date": "2018.3.29",
                "description": "Basic run of re-writed BBFWA on cec2017",
                "data": {
                    "res": "results of each run.",
                    "cost": "time cost of each run.",
                },
    }

    func_num = 30
    repetition = 50

    res = np.empty((func_num, repetition))
    cost = np.empty((func_num, repetition))

    @ray.remote
    def one_rep(i):
        def evaluate(x):
            if type(x) is np.ndarray:
                x = x.tolist()
            return cec17.eval(x, i + 1)

        model = BBFWA.BBFWA()
        model.load_prob(evaluator=evaluate,
                        dim=30,
                        max_eval=30 * 10000)
        result = model.run()
        return result

    ray.init()
    for i in tqdm(range(func_num)):
        reps_results = ray.get([one_rep.remote(i)
                                for _ in range(repetition + 1)])

        print('Prob-{}, Mean: {:<.6f}, Max: {:<.6f}, Min: {:<.6f}'.format(i + 1,
                                                                          np.mean(reps_results), np.max(reps_results), np.min(reps_results)))

    with open("logs/BBFWA_CEC17_30D.pkl", "wb") as f:
        pkl.dump([exp_info, res, cost], f)
