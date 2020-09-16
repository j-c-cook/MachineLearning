import MachineLearning.SupervisedLearning.Regression.LinearRegression as LR

import numpy as np
# from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # ---- inputs -----
    x_low = -2
    x_high = 6
    m = 50  # number of elements in feature vector

    # ------------------------------------------------------------------------------------------------------
    # get the input vector x as if we have all of the values so we can plot a line to check for over fitting
    # uniform spacing from low to high
    x_uniform = np.arange(x_low, x_high, 1/m)

    # ------------------------------------------------------------------------------------------------------

    # Question: 1
    # ------------
    # Create a input vector of random floats over a range
    # https://stackoverflow.com/questions/22071987/generate-random-array-of-floats-between-a-range
    x = np.random.uniform(low=x_low, high=x_high, size=(m, 1))
    x = np.sort(x, axis=0)
    f = np.vectorize(LR.generate_data.poly_third_order)
    y = f(x)

    # Plot x vs. y
    myplot = LR.plotting.PyPlotter()
    myplot.plot(x, y, close=False, fig_name='Question_1_Plot', show=False, plot_style='plot', line_style='',
                label='Original Data')

    # Question: 2
    # ------------
    # Part: 1
    # --------
    # Simple linear regression with "normal equations method" (non-iterative) (Maximum Likelihood)
    # Linear Hypothesis function
    phi_x = LR.features.phi_polynomial(x, order=1)
    # phi_x_all = LR.features.phi_polynomial(x_uniform, order=1)

    theta_ml = LR.features.least_squares_max_likelihood(phi_x, y)

    # get values for y predicted given theta maximum likelihood and input vector x
    y_pred = LR.features.predicted_values(theta_ml, x)
    # y_pred_all = LR.features.predict_values(phi_x_all, theta_ml)

    # get mean squared error and place it on the plot
    mse = LR.features.average_least_square_error(y_pred, y)

    myplot.plot(x, y_pred, close=False, save_fig=False, show='False', plot_style='plot',
                label='First Order Fit (MSE: {:.2f})'.format(mse))
    # myplot.plot(x, y_pred_all, close=False, save_fig=False, show='False', plot_style='plot')

    myplot.save_figure(fig_name='Question_2_Plot_Part_a')

    # Part 2:
    # -------

    phi_x = LR.features.phi_polynomial(x, order=2)

    theta_ml = LR.features.least_squares_max_likelihood(phi_x, y)

    # get values for y predicted given theta maximum likelihood and input vector x
    y_pred = LR.features.predict_values(phi_x, theta_ml)

    # get mean squared error and place it on the plot
    mse = LR.features.average_least_square_error(y_pred, y)

    myplot.plot(x, y_pred, close=False, save_fig=False, show='False',
                label='Second Order Fit (MSE: {:.2f})'.format(mse))

    myplot.save_figure(fig_name='Question_2_Plot_Part_b')

    # Part 3:
    # -------

    phi_x = LR.features.phi_polynomial(x, order=3)

    theta_ml = LR.features.least_squares_max_likelihood(phi_x, y)

    # get values for y predicted given theta maximum likelihood and input vector x
    y_pred = LR.features.predicted_values(theta_ml, x)
    from copy import deepcopy
    y_pred_no_noise = deepcopy(y_pred)

    # get mean squared error and place it on the plot
    mse = LR.features.average_least_square_error(y_pred, y)

    myplot.plot(x, y_pred, close=False, save_fig=False, show='False', label='Third Order Fit (MSE: {:.2f})'.format(mse),
                marker='*')
    myplot.save_figure(fig_name='Question_2_Plot_Part_c')

    print('theta = ' + str(theta_ml))

    # Part 4:
    # -------
    # create some noise
    # https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
    mean = 0  # mean of the normal distribution
    variance = 3  # the standard deviation of the normal distribution
    noise = np.random.normal(mean, variance, m)
    noise = noise[:, np.newaxis]

    # add some noise to the y values
    y_w_noise = y + noise

    theta_ml = LR.features.least_squares_max_likelihood(phi_x, y_w_noise)
    print("theta values for predicting on noisy data: {}".format(theta_ml))
    # get values for y predicted given theta maximum likelihood and input vector x
    y_pred = LR.features.predicted_values(theta_ml, x)

    # get mean squared error and place it on the plot
    mse = LR.features.average_least_square_error(y_pred, y)
    # compare the y_predicted fit on no noise to the y_w_noise
    mse_n = LR.features.average_least_square_error(y_pred_no_noise, y_w_noise)

    myplot_2 = LR.plotting.PyPlotter()
    # myplot_2.plot(x, y, close=False, show=False, label='Original Data')
    myplot_2.plot(x, y_w_noise, close=False, show=False, label='Original Data with Noise')
    myplot_2.plot(x, y_pred, close=False, show=False, label='Third Order Fit on Data w Noise (MSE: {:.2f})'.format(mse))
    myplot_2.plot(x, y_pred_no_noise, close=False, show=False,
                  label='Third Order Fit on Data w/o Noise (MSE: {:.2f})'.format(mse_n), plot_style='plot', marker='')

    myplot_2.save_figure(fig_name='Question_2_Plot_Part_4')

    a = 1
    # https://www.geeksforgeeks.org/implementation-of-locally-weighted-linear-regression/
    # USE THE DATA WITHOUT NOISE FOR TRAINING
    tao = 0.1
    y_test = []
    theta_vals = []
    for _x in x:
        theta, pred = LR.features.predict(x, y, _x, tao)
        a = pred.tolist()
        theta_ = theta.tolist()
        theta_vals.append(theta_[0][0])
        y_test.append(a[0][0])
    print('Theta values found for training on the data without noise LOCALLY WIEGHTED: {}'.format(theta_vals))
    y_test = np.array(y_test)
    y_test = y_test[:, np.newaxis]
    theta_vals = np.array(theta_vals)
    theta_vals = theta_vals[:, np.newaxis]

    # USE THE DATA WITH NOISE FOR TESTING
    PRED = x * theta_vals
    fix, ax = plt.subplots()

    # COMPARE PREDICTED TO NOISY
    mse = LR.features.average_least_square_error(y, y_w_noise)
    ax.scatter(x, PRED, label='Trained without noise tested with noise (MSE: {:.2f})'.format(mse))

    ax.scatter(x, y_w_noise, label='Original with noise')
    # get mean squared error and place it on the plot
    mse = LR.features.average_least_square_error(y_test, y)
    mse_1 = LR.features.average_least_square_error(y_test, y_w_noise)
    ax.scatter(x, y_test, label='Local Weighted (MSE to original: {:.2f}, MSE to noise: {:.2f})'.format(mse, mse_1))
    # get mean squared error and place it on the plot
    # mse = LR.features.average_least_square_error(y_pred, y_w_noise)
    # ax.plot(x, y_pred, label='Third Order (MSE: {:.2f})'.format(mse), ls='-')
    plt.legend(loc='upper left')
    plt.savefig('Question_1_Part_5.jpg')

    a = 1
    # --------------------------------------------------------------------------------------------------
    # Question: 3
    file_name = 'Hogsmeade_Prices.csv'
    df = pd.read_csv(file_name)
    dn = df.to_dict('list')

    keys = list(dn.keys())
    key0 = keys[0]

    inputs = []
    output = []

    for key in keys:
        if key == 'House ID':
            continue
        if 'Output:' in key:
            output.append(dn[key])
        else:
            inputs.append(dn[key])

    x = np.array(inputs).T
    y = np.array(output).T
    theta_ml = LR.features.least_squares_max_likelihood(x, y)

    y_pred = np.matmul(x, theta_ml)  # predict in one line

    mse = LR.features.average_least_square_error(y_pred, y)

    print('Mean squared error of linear regression: {}'.format(mse))

    tao = 0.1
    y_test = []
    for _x in x:
        theta, pred = LR.features.predict(x, y, _x, tao)
        a = pred.tolist()
        y_test.append(a[0][0])
    y_test = np.array(y_test)
    y_test = y_test[:, np.newaxis]

    mse = LR.features.average_least_square_error(y_test, y)

    print('Mean squared error of locally weighted: {}'.format(mse))

    x_bar = x.mean(axis=0)
    x_bar = x_bar[:, np.newaxis]
    y_bar = y.mean(axis=0)
    y_bar = y_bar[:, np.newaxis]

    x_bar_weighted = x_bar * theta_ml
    x_bar_percentage_weight = x_bar_weighted / y_bar

    x_out = x_bar_percentage_weight.tolist()

    dn_out = {'values': x_out}
    pd.DataFrame(dn_out).to_excel('file.xlsx')

    a = 1


if __name__ == '__main__':
    main()
