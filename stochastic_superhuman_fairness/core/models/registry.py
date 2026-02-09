from . import logistic, mlp, ppo, bayesian_logistic

MODEL_REGISTRY = {
    "logistic": logistic.LogisticRegressionModel,
    "mlp": mlp.MLPModel,
    "ppo": ppo.PPOModel,
    "bayesianlogistic": bayesian_logistic.BayesianLogisticRegressionModel,
}
