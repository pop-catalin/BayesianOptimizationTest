import time

import yaml
# from ray.tune.suggest.variant_generator import generate_variants as tune_generate_variants
from pydantic import BaseModel
# from ray import tune
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from bayes_opt import BayesianOptimization, UtilityFunction


# def generate_variants(config):
#     return tune_generate_variants(config)

class JobCreateRequest(BaseModel):
    priority: str
    config: str
    requested_gpu: int
    requested_gpu_memory: int
    requested_cpu: int


def create_job(priority, config, gpu, cpu):
    job_data = JobCreateRequest(
        priority=priority,
        config=config,
        requested_gpu=gpu,
        requested_gpu_memory=-1,
        requested_cpu=cpu,
    )
    return job_data


def function_to_be_optimized(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1
#maximum of this function is 1

def easy_objective(config):
    # Hyperparameters
    x, y = config["x"], config["y"]

    for step in range(config["steps"]):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = function_to_be_optimized(x, y)
        # Feed the score back back to Tune.
        tune.report(iterations=step, mean_loss=intermediate_score)
        #time.sleep(0.1)

#main function used for bayesian optimization
def bayes2():
    pbounds = {'x': (0, 20), 'y': (-100, 100)}
    optimizer = BayesianOptimization(
        f=function_to_be_optimized,
        pbounds=pbounds,
    )
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    for i in range(100):
        next_suggestion = optimizer.suggest(utility)
        target = function_to_be_optimized(**next_suggestion)
        optimizer.register(
            params=next_suggestion,
            target=target,
        )
        print(target, next_suggestion)
    print(optimizer.max)
    # optimizer.maximize(
    #     init_points=2,
    #     n_iter=100,
    # )
    #print(optimizer.max)
    # for i, res in enumerate(optimizer.res):
    #     print("Iteration {}: \n\t{}".format(i, res))
    # print(optimizer.max)

# test with bayesian optimization from ray
def bayes():
    search_space = {
        "x": (0, 20),
        "y": (-100, 100)
    }
    bayesopt = BayesOptSearch(search_space, metric="mean_loss", mode="max", utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
    for i in range(100):
        print(bayesopt.suggest(str(i)))
    # analysis = tune.run(easy_objective, search_alg=bayesopt, config={
    #         "steps": 100,
    #         "x": tune.uniform(0, 20),
    #         "y": tune.uniform(-100, 100),
    #     },metric="mean_loss", mode="min",)
    # print("Best hyperparameters found were: ", analysis.best_config)


    # for i in range(10):
    #     #bayesopt.on_trial_complete(i, res)
    #     res = bayesopt.suggest(str(i))
    #     print(res)
    #     print(bayesopt._buffered_trial_results)
    #     bayesopt.on_trial_result(i, res)
    #     for elem in bayesopt._buffered_trial_results:
    #         print(elem)
    #     print(bayesopt.optimizer._space.max())
    #     print(bayesopt.optimizer._space.res())
    #     print('---------------------------------')

def change_config_run_prefix(variables, config, stepsToSkip=0):
    name = ""
    for variable in variables:
        print(len(variable))
        temp = ''
        count = min(stepsToSkip, len(variable) - 1)
        for elem in variable:
            if count > 0:
                count -= 1
                continue
            temp += elem + '-'
        temp = temp[:-1]
        #print(type(variable))
        if '__name__' in str(variables[variable]):
            #name += variable[-1] + '=' + variables[variable]['__name__'] + '|'
            if temp:
                name += temp + '=' + variables[variable]['__name__'] + '|'
        else:
            if temp:
                name += temp + "=" + str(variables[variable]) + "|"
    print(name)
    if name:
        name = name[:-1]
        config['run_prefix'] = name
    print(len(name))
    if len(name) > 45:
        change_config_run_prefix(variables, config, stepsToSkip + 1)


def change_config_variant_name(variables, config):
    result_string = ''
    for variable in variables:
        if '__name__' in str(variables[variable]):
            #print(type(variables[variable]))
            result_string += variable[1] + '=' + variables[variable]['__name__'] + '-'
            #asdf = variables[variable]['__name__']
            #print(variable)
            #print(asdf)
            print(variables[variable])
    print(result_string)

if __name__ == '__main__':
    bayes2()