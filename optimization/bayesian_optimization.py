from simulator.scenario_runner import run_scenario
from simulator.scenario_runner import generate_scenarios
import optuna
import multiprocessing as mp
import pandas as pd

NUM_SCENARIOS = 2  # Can increase to 5 once you have a few good layouts

SCENARIOS = generate_scenarios(NUM_SCENARIOS)

def objective(trial):
    # Suggest parameters to optimize
    alpha1 = trial.suggest_uniform('alpha1', 5.0, 20.0)
    alpha2 = trial.suggest_uniform('alpha2', 0.0, 1.0)
    vel_weight = trial.suggest_uniform('vel_weight', 0.1, 1.0)
    avoidance = trial.suggest_uniform('avoidance', 1.0, 10.0)
    attraction = trial.suggest_uniform('attraction', 1.0, 30.0)

    scores = []
    dummy_queue = mp.Manager().Queue()
    for scenario in range(NUM_SCENARIOS):
        # run_scenario returns the score for one scenario
        score = run_scenario(
            scenario,
            dummy_queue,
            alpha1=alpha1,
            alpha2=alpha2,
            vel_weight=vel_weight,
            avoidance=avoidance,
            attraction=attraction,
            positions=SCENARIOS[scenario]
        )
        scores.append(score)

    # maximize average score across scenarios
    return sum(scores) / len(scores)

def run_optimization(n_trials=50, csv_path="results.csv"):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)

    print('Best parameters:', study.best_params)
    print('Best average score:', study.best_value)