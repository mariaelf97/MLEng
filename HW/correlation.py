import warnings

import numpy as np
import pandas as pd
import scipy.stats as ss
from define_data_type import define_cat


def get_correlation(
    dataset, predictor1, predictor2, bias_correction=False, tschuprow=True
):
    predictor1_cat = define_cat(dataset, predictor1)
    predictor2_cat = define_cat(dataset, predictor2)
    if predictor1_cat & predictor2_cat:
        corr_coeff = np.nan
        try:
            crosstab_matrix = pd.crosstab(dataset[predictor1], dataset[predictor2])
            n_observations = crosstab_matrix.sum().sum()

            yates_correct = True
            if bias_correction:
                if crosstab_matrix.shape == (2, 2):
                    yates_correct = False

            chi2, _, _, _ = ss.chi2_contingency(
                crosstab_matrix, correction=yates_correct
            )
            phi2 = chi2 / n_observations

            # r and c are number of categories of x and y
            r, c = crosstab_matrix.shape
            if bias_correction:
                phi2_corrected = max(
                    0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1)
                )
                r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
                c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
                if tschuprow:
                    corr_coeff = np.sqrt(
                        phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                    )
                    return corr_coeff
                corr_coeff = np.sqrt(
                    phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
                )
                return corr_coeff
            if tschuprow:
                corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
                return corr_coeff
            corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
            return corr_coeff
        except Exception as ex:
            print(ex)
            if tschuprow:
                warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
            else:
                warnings.warn("Error calculating Cramer's V", RuntimeWarning)
            return corr_coeff
    elif (predictor1_cat & (not predictor2_cat)) | (
        predictor2_cat & (not predictor1_cat)
    ):
        if predictor1_cat:
            categories = dataset[predictor1].to_numpy()
            measurements = dataset[predictor2].to_numpy()
        else:
            categories = dataset[predictor2].to_numpy()
            measurements = dataset[predictor1].to_numpy()

        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0, cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
        numerator = np.sum(
            np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
        )
        denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = np.sqrt(numerator / denominator)
        return eta

    else:
        corr = np.corrcoef(dataset[predictor1], dataset[predictor2])[0, 1]
        return corr
