# statistics functions

'''
Python file containing list of functions useful for conducting statistical analysis of data sets. 

'''
import numpy as np 
import scipy.stats as ss
from scipy import stats
import matplotlib.pyplot as plt


#### OUTLIER TREATMENT ####
def filter_outliers_by_std(dataframe, parameter, num_sigmas=3, use_median=False):
    '''
    Removes outliers from a DataFrame based on the specified number of standard deviations from the mean or median.
    
    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the data.
    parameter (str): The column name of the dataframe to analyze.
    num_sigmas (int), optional: The number of standard deviations to use as the threshold for defining outliers (default is 3).
    use_median (bool), optional: Whether to use the median instead of the mean for the central tendency measure (default is False).
    '''
    if use_median:
        center = np.median(dataframe[parameter])
    else:
        center = np.mean(dataframe[parameter])
    
    std = np.std(dataframe[parameter])
    lower_bound = center - (num_sigmas * std) 
    upper_bound = center + (num_sigmas * std)

    filtered_df = dataframe[(dataframe[parameter] < upper_bound) & (dataframe[parameter] > lower_bound)]
    
    return filtered_df


def create_outlier_mask(array, num_sigmas=4):
    '''
    Generates a boolean mask that identifies values within a specified number of standard deviations from the mean.

    Parameters:
    array (array-like): Array of numeric values from which to identify outliers.
    num_sigmas (int), optional: The number of standard deviations to use for defining the range (default is 4).
    '''
    mean = np.mean(array)
    std = np.std(array)
    lower_bound = mean - (num_sigmas * std)
    upper_bound = mean + (num_sigmas * std)

    mask = (array > lower_bound) & (array < upper_bound)
    return mask



#### RESIDUAL METHODS ####
def plot_binned_data(x, y, x_label = None, y_label=None, num_bins=None):
    '''
    Plots relational scatter plot between two parameters.
    Parameters must be in form of Numpy array or Pandas Series
    '''
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy) # helps to plot density 

    if num_bins == None: 
        bin_medians, bin_edges, binnumber = stats.binned_statistic(x=x, values=y,statistic= 'median')
        std, s_edges, s_binnumber = stats.binned_statistic(x=x, values = y, statistic = 'std')
    
    else: 
        bin_medians, bin_edges, binnumber = stats.binned_statistic(x=x, values=y,statistic= 'median', bins = num_bins)
        std, s_edges, s_binnumber = stats.binned_statistic(x=x, values = y, statistic = 'std', bins = num_bins)

    plt.figure(figsize =(8,6))
    plt.scatter(x, y, c = z, alpha = 0.3)
    plt.xlabel(x_label if x_label else 'X')
    plt.ylabel(y_label if x_label else 'Y')

    # Errorbars 
    errbars = (bin_edges[1:] + bin_edges[:-1])/2        # x position of median points
    plt.errorbar(errbars ,bin_medians, yerr=std, fmt='r-')  # This plots median point x value, and median (which is y value)
    plt.show()


def residuals(x, y, num_bins=None): 
    '''
    Takes a parameter y and decouples parameter x from it (finding the residuals), 
    returning the Delta Parameter (the y parameter with its correlation to x removed).
    '''

    # Statistics calculations 
    if num_bins == None:
        bin_medians, bin_edges, binnumber = stats.binned_statistic(x=x, values=y,statistic= 'median')
        std, s_edges, s_binnumber = stats.binned_statistic(x=x, values = y, statistic = 'std')
        num_bins = len(bin_edges) - 1
    else: 
        bin_medians, bin_edges, binnumber = stats.binned_statistic(x=x, values=y,statistic= 'median', bins = num_bins)
        std, s_edges, s_binnumber = stats.binned_statistic(x=x, values = y, statistic = 'std', bins = num_bins)
    errbars = (bin_edges[1:] + bin_edges[:-1])/2 

    # Slope calculations 
    x1 = errbars[:-1] ; x2 = errbars[1:]
    y1 = bin_medians[:-1] ; y2 = bin_medians[1:]
    m = (y2 - y1) / (x2 - x1) # slope
    b = -m*x1 + y1            # y-intercept

    # Delta Parameter Calculation 
    y_exp = np.zeros(len(y))  # y_expected array
    i = 0
    while i < len((x)):
        xi = x[i]
        for n in range(num_bins-1): # had to be 1 less than 3 of bins
            if xi <= errbars[n] and xi >= errbars[n-1]:
                Y = m[n] * xi + b[n]
                y_exp[i] = Y 
                deltaParam = y - y_exp
            else:
                pass
        i+=1
    
    residuals = y - y_exp

    return deltaParam, y_exp, residuals 


def residuals_masked(x, y, num_bins=None): 
    '''Takes a parameter in and decouples mass from it, returning the Delta Parameter. 
        Treats the delta parameter outliers using standard dev. of 4 and higher and masking.
        x and y must be numpy arrays.
        '''
    deltaParam, y_exp, residuals = residuals_masked(x, y, num_bins=None)

    dmask = create_outlier_mask(deltaParam)

    delta_param = deltaParam[dmask]
    yexp = y_exp[dmask]
    res = residuals[dmask]
    X = x[dmask] # the original "X" array (which has typically been logmstar); sliced by dmask 
 
    return delta_param, y_exp, res, X, dmask 



#### RESAMPLING METHODS ####

def get_lin_params(x,y):    # courtesy of Janvi!
    '''
    Fits a linear polynomial to the given x and y data points and calculates the Pearson correlation coefficient as well as its p-value.
    Utilizies numpy and scipy.stats for calculating Pearson correlation. 

    Parameters:
    x (array-like): x data points.
    y (array-like): y data points corresponding to x.

    Returns:
    xp (np.ndarray): 100 evenly spaced x values covering the original range of x.
    p (np.poly1d): Polynomial object which evaluates the polynomial using the fitted coefficients.
    Rval (float): Pearson's correlation coefficient indicating the linear relationship strength.
    pval (float): Two-tailed p-value for a hypothesis test whose null hypothesis is that the slope is zero.
    '''
    z = np.polyfit(x, y,1)
    # converts polynomial coeff's into object 'p' that can be called as a function to evaluate polynomial at given x value
    p = np.poly1d(z) 
    xp = np.linspace(min(x),max(x),100) # to plot linear fit line over the original data range
    Rval,pval = ss.pearsonr(x, y) # gets Pearson coefficient (R-value) and p-value
    
    return xp,p,Rval,pval



def bootstrap_linear_fit(x, y, n_bootstrap=10000, standard_devs=2):
    '''
    Performs bootstrap resampling to estimate the uncertainty in linear polynomial fitting parameters,
    along with the uncertainty in the Pearson correlation coefficient and its p-value, over multiple
    samples drawn with replacement.
    Utilizes 'get_lin_params' function to fit models and calculate statistics for each bootstrap sample. 

    Parameters:
    x (array-like): x data points.
    y (array-like): y data points corresponding to x.
    n_bootstrap (int, optional): Number of bootstrap samples to generate. Default is 10,000.
    standard_dev (int, optional): Number of standard deviations that should be returned for plotting error envelope. 

    Returns:
    xp (np.ndarray): 100 evenly spaced x values covering the original range of x.
    lower_lim (np.ndarray): The lower limit *standard_devs away from the mean of the returned points. 
    upper_lim (np.ndarray): The upper limit *standard_devs away from the mean of the returned points. 
    rvals_stats (tuple): Standard deviation and mean of Pearson's correlation coefficients from the bootstrap samples.
    pvals_stats (tuple): Standard deviation and mean of the p-values from the bootstrap samples.
    fits (np.ndarray): 2D array of all polynomial evaluations at points in xp for each bootstrap sample, used to calculate bounds (for error envelope).
    '''
    fits = []
    rvals = []
    pvals =[]
    xp = np.linspace(np.min(x), np.max(x), 100)
    
    # Bootstrap resampling
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(np.arange(len(x)), size=len(x), replace=True)
        x_sample, y_sample = x[indices], y[indices]
        
        # Fit linear model to bootstrap sample
        _, p, Rval, pval = get_lin_params(x_sample, y_sample)   
        fits.append(p(xp))
        rvals.append(Rval)
        pvals.append(pval)
    
    # Convert list of arrays into a 2D numpy array 
    fits = np.array(fits)
    
    # Determine high and low bounds for each point along xp -- to create Error Envelope
    #low_bound = np.min(fits, axis=0)
    #high_bound = np.max(fits, axis=0)

    # Find standard deviation and mean of each coefficient - to plot error envelope
    N = len(fits[0])
    mus = np.zeros(N)
    stds = np.zeros(N)

    for i in range(N):
        mus[i] = np.mean(fits[:,i])
        stds[i] = np.std(fits[:,i])

    upper_lim = mus + standard_devs*stds # 1 standard deviation 
    lower_lim = mus - standard_devs*stds
    
    return xp, lower_lim, upper_lim, (np.std(rvals), np.mean(rvals)), \
                                        (np.std(pvals), np.mean(pvals)), fits



def jackknife_linear_fit(x, y, standard_devs=2):
    '''
    Performs jackknife resampling to estimate the uncertainty in the parameters of a linear fit,
    as well as the Pearson correlation coefficient and its p-value, by systematically omitting
    one data point at a time from the dataset.
    Utilizes 'get_lin_params' function to fit linear models to the jackknife samples. 

    Parameters:
    x (array-like): x data points.
    y (array-like): y data points corresponding to x.
    standard_devs (int, optional): Number of standard deviations that should be returned for plotting error envelope. 

    Returns:
    xp (np.ndarray): 100 evenly spaced x values covering the original range of x.
    lower_lim (numpy.ndarray): Lower boundary of the error envelope for the fitted polynomial at each point in xp.
    upper_lim (numpy.ndarray): Upper boundary of the error envelope for the fitted polynomial at each point in xp.
    rvals_stats (tuple): Standard deviation and mean of Pearson's correlation coefficients from the jackknife samples.
    pvals_stats (tuple): Standard deviation and mean of the p-values from the jackknife samples.
    fits (np.ndarray): 2D array of all polynomial evaluations at points in xp for each jackknife sample, used to calculate bounds (for error envelope).
    '''
    n = len(x)
    fits = []
    rvals = []
    pvals =[]
    
    xp = np.linspace(np.min(x), np.max(x), 100)

    for i in range(n):
        # First exclude the i-th data point
        x_jack = np.delete(x, i)
        y_jack = np.delete(y, i)
        
        # Fit linear model to the jackknife sample
        xp, p, Rval, pval = get_lin_params(x_jack, y_jack)
        fits.append(p(xp))
        rvals.append(Rval)
        pvals.append(pval)

    fits = np.array(fits)
    
    # Determine high and low bounds for each point along xp
    low_bound = np.min(fits, axis=0)
    high_bound = np.max(fits, axis=0)

    # Find standard deviation and mean of each coefficient - to plot error envelope
    N = len(fits[0])
    mus = np.zeros(N)
    stds = np.zeros(N)

    for i in range(N):
        mus[i] = np.mean(fits[:,i])
        stds[i] = np.std(fits[:,i])

    upper_lim = mus + standard_devs*stds # 1 standard deviation 
    lower_lim = mus - standard_devs*stds
    
    return xp, low_bound, high_bound, (np.std(rvals), np.mean(rvals)), \
                                        (np.std(pvals), np.mean(pvals)), fits
