# statistics functions

'''
Python file containing list of functions useful for conducting statistical analysis of data sets. 

'''
import numpy as np 
import scipy.stats as ss


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
