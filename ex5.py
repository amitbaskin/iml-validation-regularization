import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression


LOW = -3.2
HIGH = 2.2
SIZE = 1500
TRAIN_FACTOR = 1000
MAX_DEGREE = 15
FOLDS_AMOUNT = 5
DEGS = range(1, MAX_DEGREE + 1)
FIRST_ITERATION_SIGMA = 1
SECOND_ITERATION_SIGMA = 5
RIDGE_PARAMS = np.linspace(0, 0.05, 1000)
LASSO_PARAMS = np.linspace(0.01, 1, 1000)


def default_labels_func():
    """
    The labels function for the samples of question four
    :param sigma: The standard deviation of which to create noise from
    :return: The function
    """
    return lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)



def generate_samples(low, high, size):
    """
    Generates samples uniformly from an interval
    :param low: The lower bound of the interval
    :param high: The upper bound of the interval
    :param size: The amount of samples to generate
    :return: The samples
    """
    return np.random.uniform(low, high, size)


def generate_labels(f, all_samples):
    """
    Generates the labels of all the samples that were already generated
    :param f: The label function
    :param all_samples: The samples to get their labels
    :return: The labels
    """
    return np.apply_along_axis(f, 0, all_samples)


def add_noise(labels, sigma, mean=0):
    return labels + np.random.normal(mean, sigma, len(labels))


def generate_data(sigma):
    """
    Generates the data for question four
    :param sigma: The standard deviation for the noise of the labels
    :return: Train samples, train labels, test samples, test labels
    """
    X = generate_samples(LOW, HIGH, SIZE)
    y = generate_labels(default_labels_func(), X)
    y = add_noise(y, sigma)
    D, T = split_data(X, y, TRAIN_FACTOR)
    all_train_samples = D[0]
    all_train_samples = all_train_samples.reshape(-1, 1)
    all_train_labels = D[1]
    test_samples = T[0]
    test_samples = test_samples.reshape(-1, 1)
    test_labels = T[1]
    return all_train_samples, all_train_labels, test_samples, test_labels


def split_data(all_samples, all_labels, train_factor):
    """
    Splits the data into train set and test set
    :param all_samples: The samples to split
    :param all_labels: The labels to split
    :param train_factor: A number which specifies how many samples should
    the train set contain
    :return: The train set and test set
    """
    train_set = all_samples[:train_factor], all_labels[:train_factor]
    test_set = all_samples[train_factor:], all_labels[train_factor:]
    return train_set, test_set


def get_k_folds(k, train_samples, train_labels):
    """
    Splits the train samples into folds
    :param k: The amount of folds to split the data into
    :param train_samples: The samples to split
    :param train_labels: The labels to split
    :return: A list of the folds of the samples, where each fold should play
    the role of the validation set, and also returns a list of the samples
    where the i'th item consists of all the samples but the i'th fold,
    which should play the role of a train set that matches the i'th fold as
    the validation set"""
    total_amount = len(train_samples)
    amount_per_fold = total_amount // k
    validation_excluded = []
    folds = []

    for i in range(1, k + 1):
        a = train_samples[:(i - 1) * amount_per_fold]
        b = train_samples[i * amount_per_fold:]
        curr_samples = np.concatenate((a, b))
        a = train_labels[:(i - 1) * amount_per_fold]
        b = train_labels[i * amount_per_fold:]
        curr_labels = np.concatenate((a, b))
        validation_excluded.append((curr_samples, curr_labels))

    counter = 0
    for i in range(k):
        fold_samples = train_samples[counter:counter + amount_per_fold]
        fold_labels = train_labels[counter:counter + amount_per_fold]
        folds.append((fold_samples, fold_labels))
        counter += amount_per_fold

    return folds, validation_excluded


def two_folds_get_single_model(model, train_samples, train_labels,
                               validation_samples, validation_labels):
    """
    Gets a single model for the two folds version
    :param model: The model to train
    :param train_samples: Train samples
    :param train_labels: Train labels
    :param validation_samples: Validation samples
    :param validation_labels: Validation labels
    :return: The trained model, its train error and its validation error
    """
    model.fit(train_samples, train_labels)
    y_hat_train = model.predict(train_samples)
    train_error = np.mean((train_labels - y_hat_train) ** 2)
    y_hat_validation = model.predict(validation_samples)
    validation_error = np.mean((validation_labels - y_hat_validation) ** 2)

    return model, train_error, validation_error


def poly_generator(deg):
    """
    A generator for a polynomial regression learner
    :param deg: The degree of the polynomial to generate
    :return: The generated learner
    """
    return make_pipeline(PolynomialFeatures(deg), LinearRegression())


def two_folds_get_all_models(model_generator, params, train_samples,
                             train_labels, validation_samples,
                             validation_labels):
    """
    Gets all the models of the two folds case
    :param model_generator: The model generator
    :param params: The parameters from which to generate the learners
    :param train_samples: Train samples
    :param train_labels: Train labels
    :param validation_samples: Validation samples
    :param validation_labels: Validation labels
    :return: The models
    """
    models = []
    for param in params:
        model = model_generator(param)
        model = two_folds_get_single_model(
            model, train_samples, train_labels, validation_samples,
            validation_labels)
        models.append(model)
    return models


def multiple_folds_get_single_model(model, folds, validation_excluded):
    """
    Gets a single model from the multiple folds case
    :param model: The model to train
    :param folds: The validation sets
    :param validation_excluded: The train sets
    :return: The trained model, its mean train error and its mean validation
    error
    """
    folds_amount = len(folds)
    train_losses = []
    validation_losses = []
    for i in range(folds_amount):
        train_samples = validation_excluded[i][0]
        train_labels = validation_excluded[i][1]
        model.fit(train_samples, train_labels)

        y_hat = model.predict(train_samples)
        y = train_labels
        train_losses.append(np.mean((y - y_hat) ** 2))

        fold_samples = folds[i][0]
        y_hat = model.predict(fold_samples)
        y = folds[i][1]
        validation_losses.append(np.mean((y - y_hat) ** 2))

    train_error = np.mean(train_losses)
    validation_error = np.mean(validation_losses)

    return model, train_error, validation_error


def multiple_folds_get_all_models(model_generator, params, folds,
                                  validation_excluded):
    """
    Gets all the models for the multiple folds case
    :param model_generator: The model generator
    :param params: The parameters from which to generate the models
    :param folds: The validation sets
    :param validation_excluded: The train sets
    :return: The trained models as triplets, each one consists of the model,
    its mean train error and its mean validation error
    """
    models = []
    for param in params:
        model = model_generator(param)
        poly = multiple_folds_get_single_model(model, folds,
                                               validation_excluded)
        models.append(poly)
    return models


def get_best_model(models):
    """
    Gets the best model (triplet)
    :param models: The models to choose from
    :return: The best model (triplet)
    """
    return min(models, key=lambda poly: poly[2])


def get_errors(models):
    """
    Gets a list of the mean train errors and a list of the mean validation
    errors of the given models
    :param models: The models to get their errors
    :return: The list of train errors and the list of validation errors
    """
    train_errors = []
    validations_errors = []
    for model in models:
        train_errors.append(model[1])
        validations_errors.append(model[2])
    return train_errors, validations_errors


def get_multiple_folds_models(model_generator, params,
                              all_train_samples,
                              all_train_labels):
    """
    Gets the models for the multiple folds case
    :param model_generator: The model generator
    :param params: The parameters from which to generate the models
    :param all_train_samples: Train samples
    :param all_train_labels: Train labels
    :return: The models
    """
    S, V = get_k_folds(FOLDS_AMOUNT, all_train_samples, all_train_labels)
    return multiple_folds_get_all_models(model_generator, params, S, V)


class Data:
    """
    Data object for question four
    """
    def __init__(self, sigma):
        """
        Initializes a data object for question four according to the noise
         of the labels
        :param sigma: The noise to give to the labels
        """
        self.all_train_samples, self.all_train_labels, self.test_samples, \
        self.test_labels = generate_data(sigma)


def fourth_question_items_b_c(all_train_samples, all_train_labels,
                              param_getter, param_description,
                              regression_description):
    """
    Runs a general version of items b and c of question four
    :param all_train_samples: Train samples
    :param all_train_labels: Train labels
    :param param_getter: Gets the parameter from the model
    :param param_description: Description of the parameter
    :param regression_description: Description of the regression
    :return: None
    """
    S, V = get_k_folds(2, all_train_samples, all_train_labels)
    S, V = S[0], V[0]
    train_samples = S[0]
    train_labels = S[1]
    validation_samples = V[0]
    validation_labels = V[1]
    models = two_folds_get_all_models(poly_generator, DEGS, train_samples,
                                      train_labels, validation_samples,
                                      validation_labels)
    best_model = get_best_model(models)
    best_param = param_getter(best_model)
    print()
    print('  The best {} with two folds for {} is: '.
          format(param_description, regression_description), best_param)


def fourth_question_items_d_e(model_generator, params, all_train_samples,
                              all_train_labels, xlabel, title1, title2,
                              title3, param_getter, param_description,
                              regression_description):
    """
    Runs a general version of items d and e of question four
    :param model_generator: The model generator
    :param params: The parameters from which to generate the models
    :param all_train_samples: Train samples
    :param all_train_labels: Train labels
    :param xlabel: The labels for the X axis
    :param title1: One part of the title for the graph
    :param title2: Second part of the title for the graph
    :param title3: Third part of the title for the graph
    :param param_getter: Gets the parameter from the model
    :param param_description: Description of the parameter
    :param regression_description: Description of the regression
    :return: None
    """
    models = get_multiple_folds_models(model_generator, params,
                                       all_train_samples,
                                       all_train_labels)

    best_model = get_best_model(models)
    best_param = param_getter(best_model)
    print()
    print('  The best {} with cross validation for {} is: '.
          format(param_description, regression_description), best_param)

    train_errors, validation_errors = get_errors(models)

    plt.plot(params, train_errors, linewidth=6, alpha=0.5, label='Train '
                                                                  'Errors')
    plt.plot(params, validation_errors, 'r-', linewidth=2,
             label='Validation Errors')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(xlabel)
    plt.ylabel('Error')
    plt.title('The train and validation errors\nof {}\nas a function '
              'of {}\n{}'.format(title1, title2, title3))
    plt.show()


def fourth_question_items_f_g(model_generator, all_train_samples,
                              all_train_labels,
                              test_samples, test_labels, best_param):
    """
    Runs a general version of items f and g of question four
    :param model_generator: The model generator
    :param all_train_samples: Train samples
    :param all_train_labels: Train labels
    :param test_samples: Test samples
    :param test_labels: Test labels
    :param best_param: The parameter of the best model
    :return: None
    """
    model = model_generator(best_param)
    model.fit(all_train_samples, all_train_labels)

    y_hat = model.predict(all_train_samples)
    y = all_train_labels
    train_error = np.mean((y - y_hat) ** 2)
    y_hat = model.predict(test_samples)
    y = test_labels

    test_error = np.mean((y - y_hat) ** 2)
    diff = np.fabs(train_error - test_error)

    print('  The train error is: {}'.format(train_error))
    print('  The test error is: {}'.format(test_error))
    print('  The difference between the train error and the test error is: '
          '{}'.format(diff))


def run_all_items(model_generator, params, xlabel, title1, title2, title3,
                  best_param, all_train_samples, all_train_labels,
                  test_samples, test_labels, param_getter, param_description,
                  regression_description):
    """
    Runs a general version of all items from question four
    :param model_generator: The model generator
    :param params: The parameters from which to generate the models
    :param xlabel: The label of the X axis
    :param title1: One part of the title for the graph
    :param title2: Second part of the title for the graph
    :param title3: Third part of the title for the graph
    :param best_param: The parameter of the best model
    :param all_train_samples: Train samples
    :param all_train_labels: Train labels
    :param test_samples: Test samples
    :param test_labels: Test labels
    :return: None
    """
    fourth_question_items_b_c(all_train_samples, all_train_labels,
                              param_getter, param_description,
                              regression_description)
    fourth_question_items_d_e(model_generator, params, all_train_samples,
                              all_train_labels, xlabel, title1, title2,
                            title3, param_getter, param_description,
                              regression_description)
    fourth_question_items_f_g(model_generator, all_train_samples,
                              all_train_labels,
                              test_samples, test_labels, best_param)


def get_deg_from_poly_model(poly):
    """
    Gets the degree of a given polynomial regression learner
    :param poly: A triplet of a polynomial regression learner
    :return: The degree of the polynomial regression learner
    """
    return poly[0].named_steps['polynomialfeatures'].degree


def run_question_four(sigma):
    data = Data(sigma)

    all_train_samples, all_train_labels, test_samples, test_labels = \
        data.all_train_samples, data.all_train_labels, data.test_samples, \
        data.test_labels

    models = get_multiple_folds_models(poly_generator, DEGS,
                                       all_train_samples,
                                       all_train_labels)

    best_deg = get_deg_from_poly_model(get_best_model(models))

    run_all_items(poly_generator, DEGS, 'Degree',
                  'polynomials in polynomial regression,',
                  'as a function of the degree',
                  'samples generated from a normal distribution\nwith mean '
                  'with value of zero\nand standard deviation of {}'.format(
                      sigma),
                  best_deg, all_train_samples, all_train_labels,
                  test_samples, test_labels, get_deg_from_poly_model,
                  'degree', 'polynomial regression on the given data')


def get_diabetes_data():
    X, y = datasets.load_diabetes(return_X_y=True)
    train_samples = X[:50]
    train_labels = y[:50]
    test_samples = X[50:]
    test_labels = y[50:]
    return train_samples, train_labels, test_samples, test_labels


def ridge_generator(alpha):
    """
    A generator for a Ridge regression learner
    :param alpha: The regularization parameter of the learner we want to
    generate
    :return: The generated learner
    """
    return Ridge(alpha)


def lasso_generator(alpha):
    """
    A generator for a Lasso regression Learner
    :param alpha: The regularization parameter of the learner we want to
    generate
    :return: The generated learner
    """
    return Lasso(alpha, max_iter=10**4)


def get_best_reg_param(model):
    return model[0].alpha


def run_question_five():
    """
    Run question five
    :return: None
    """
    train_samples, train_labels, test_samples, test_labels = get_diabetes_data()

    models = get_multiple_folds_models(ridge_generator, RIDGE_PARAMS,
                                       train_samples,
                                       train_labels)
    ridge_best_param = get_best_reg_param(get_best_model(models))
    ridge_best_model = ridge_generator(ridge_best_param)
    ridge_best_model.fit(train_samples, train_labels)
    ridge_y_hat = ridge_best_model.predict(test_samples)
    test_error = np.mean((test_labels - ridge_y_hat) ** 2)
    print()
    print('  The test error for the best Ridge model is: {}'.format(test_error))

    linear_regression_model = ridge_generator(0)
    linear_regression_model.fit(train_samples, train_labels)
    linear_regression_model_y_hat = linear_regression_model.predict(test_samples)
    test_error = np.mean((test_labels - linear_regression_model_y_hat) ** 2)
    print('  The test error for a linear regression model is: {}'.format(
        test_error))

    models = get_multiple_folds_models(lasso_generator, LASSO_PARAMS,
                                       train_samples,
                                       train_labels)
    lasso_best_param = get_best_reg_param(get_best_model(models))
    lasso_best_model = lasso_generator(lasso_best_param)
    lasso_best_model.fit(train_samples, train_labels)
    lasso_y_hat = lasso_best_model.predict(test_samples)
    test_error = np.mean((test_labels - lasso_y_hat) ** 2)

    print('  The test error for the best Lasso model is: {}'.format(test_error))
    fourth_question_items_d_e(ridge_generator, RIDGE_PARAMS, train_samples,
                              train_labels,
                              'regularization parameter',
                              'Ridge regression models',
                            'the regularization parameter', '',
                              get_best_reg_param, 'regularization parameter',
                              'Ridge regression on the given data')

    fourth_question_items_d_e(lasso_generator, LASSO_PARAMS, train_samples,
                              train_labels,
                              'regularization parameter',
                              'Lasso regression models',
                            'the regularization parameter', '',
                              get_best_reg_param, 'regularization parameter',
                              'Lasso regression on the given data')


# run_question_four(FIRST_ITERATION_SIGMA)
# run_question_four(SECOND_ITERATION_SIGMA)
# run_question_five()
