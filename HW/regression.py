import statsmodels.api as sm


def logistic_reg(dataset, predictor, response):
    y = dataset[response]
    x = dataset[predictor]
    cons = sm.add_constant(x)
    logistic_regression_model = sm.Logit(y, cons)
    logistic_regression_model_fitted = logistic_regression_model.fit()
    # Get the stats
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])
    return p_value, t_value
