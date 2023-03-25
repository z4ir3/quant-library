import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
# import seaborn as sns
# import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF 
from scipy.optimize import minimize
# from numpy.linalg import inv
from math import gamma 
from scipy import stats, integrate
from pandas_datareader import data
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
plt.style.use("seaborn-dark") 

# Time Series

def getassets(tickers, 
              startdate: str = "2011-12-31", 
              enddate: str   = "2022-12-31", 
              datatype: str  = "Adj Close",
              dsource: str   = "yahoo",
              interval: str  = "1d"):
    '''
    '''
    try:
        # Firstly, try with pandas datareader
        assets = yf.download(
            tickers = tickers,
            start = startdate,
            end = enddate,
            interval = interval,
            progress = False
        )
        if len(tickers) > 1:
            assets.index = assets.index.tz_localize(None)          # change yf date format to match pandas datareader
            assets = assets.filter(like = datatype)                # reduce to just selected columns
            assets.columns = assets.columns.get_level_values(1)    # tickers as col names
        else:
            assets = assets.filter(like = datatype)                # reduce to just selected columns
            assets = assets.rename(columns={datatype:tickers[0]})  # tickers as col names
    except:
        print("YF-ERROR: cannot download data using yahoo-finance")
        print("Trying with Pandas DataReader")
        syy, smm, sdd = startdate.split("-")
        eyy, emm, edd = enddate.split("-")
        assets = pd.DataFrame()
        for i,asset_name in enumerate(tickers):
            print("- Loading {} ({:.0f}/{:.0f})\t".format(asset_name,i+1,len(tickers)))
            assets[asset_name] = data.DataReader(
                asset_name, 
                data_source = dsource,
                start = datetime(int(syy),int(smm),int(sdd)), 
                end = datetime(int(eyy),int(emm),int(edd))
            )[datatype]
    
    return assets

#### Returns

def compound(s) -> pd.Series or pd.DataFrame:
    '''
    Single compound rule for a pd.Dataframe or pd.Series of returns. 
    In the former case, the method compounds the returns for every column (Series) by using pd.aggregate.
    The method returns a single number using prod(). 
    Note that this is equivalent to (but slower than): np.expm1( np.logp1(s).sum() )
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(compound)
    elif isinstance(s, pd.Series):
        return (1 + s).prod() - 1
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def compound_returns(
    s, 
    start = 1
    ):
    '''
    Compound a pd.Dataframe or pd.Series of returns from an inputi nitial start value.
    In the former case, the method compounds the returns for every column (Series) by using pd.aggregate. 
    The method returns a pd.Dataframe or pd.Series using cumprod(). 
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(compound_returns, start=start)
    elif isinstance(s, pd.Series):
        return start * (1 + s).cumprod()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def compute_returns(
    s, 
    mpor      = 1, 
    ascending = True, 
    dropna    = False
    ):
    '''
    Computes the arithmetic returns of a pd.Dataframe or pd.Series of prices 
    In the former case, it computes the returns for every column (Series) by using pd.aggregate.
    The method returns a pd.Dataframe or pd.Series using shift().
    Default MPOR value = 1.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compute_returns, mpor=mpor, ascending=ascending, dropna=dropna)
    elif isinstance(s, pd.Series):
        if ascending:
            r = s / s.shift(mpor) - 1
        else:
            r = s / s.shift(-mpor) - 1
        if dropna:
            return r.dropna()
        else:
            return r
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def compute_logreturns(
    s, 
    mpor      = 1, 
    ascending = True, 
    dropna    = False
    ):
    '''
    Computes the log-returns returns of a pd.Dataframe or pd.Series of prices.
    In the former case, it computes the returns for every column (Series) by using pd.aggregate.
    The method returns a pd.Dataframe or pd.Series using shift().
    Default MPOR value = 1.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compute_logreturns, mpor=mpor, ascending=ascending, dropna=dropna)
    elif isinstance(s, pd.Series):
        if ascending:
            r = np.log( s / s.shift(mpor) )
        else:
            r = np.log( s / s.shift(-mpor) )
        if dropna:
            return r.dropna()
        else:
            return r
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def annualize_returns(
    s, 
    ppy
    ):
    '''
    Computes the annualized returns (returns-per-year) of a pd.Dataframe or pd.Series of returns.
    The variable ppy can be, for example
    12 for weekly, 
    52 for monthly, 
    252 for daily
    data returns. 
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(annualize_returns, ppy=ppy)
    elif isinstance(s, pd.Series):
        growth = (1 + s).prod()
        return growth**(ppy/s.shape[0]) - 1
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

#### Downside Risk Measures
        
def drawdown(
    s, 
    rets      = False, 
    maxd      = True,
    percent   = True, 
    ascending = True
    ):
    '''
    Computes the drawdown of a pd.Dataframe or pd.Series of prices.
    - rets = True, the input DataFrame/Series consists of returns (prices obtained by compounding)
      rets = False, the input DataFrame/Series consists of prices
    - maxd = True, returns the maximum drawdown
      maxd = False, returns the drawdown series
    - percent = True, for (relative) drawdown as a percentage of the maximums 
      percent = False, for (absolute) drawdown 
    - ascending = True, if the input DataFrame/Series is sorted in ascending order.
      ascending = False, if the input DataFrame/Series is sorted in descending order.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(drawdown, rets=rets, maxd=maxd, percent=percent, ascending=ascending)
    elif isinstance(s, pd.Series):
        if not ascending:
            s = s.sort_index(ascending=True)
        if rets:
            s = compound_returns(s, start=1)
            if percent is False:
                percent = True
        MM = s.cummax()
        if percent:
            ddown = (s - MM) / MM
        else:
            ddown = s - MM
        if maxd:
            return ddown.min()
        else:
            return ddown
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
        
def drawup(
    s, 
    rets      = False, 
    maxd      = True,
    percent   = True, 
    ascending = True
    ):
    '''
    Computes the drawup of a pd.Dataframe or pd.Series of prices.
    - rets = True, the input DataFrame/Series consists of returns (prices obtained by compounding)
      rets = False, the input DataFrame/Series consists of prices
    - maxd = True, returns the maximum drawup
      maxd = False, returns the drawup series
    - percent = True, for (relative) drawup as a percentage of the maximums 
      percent = False, for (absolute) drawup 
    - ascending = True, if the input DataFrame/Series is sorted in ascending order.
      ascending = False, if the input DataFrame/Series is sorted in descending order.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(drawup, rets=rets, maxd=maxd, percent=percent, ascending=ascending)
    elif isinstance(s, pd.Series):
        if not ascending:
            s = s.sort_index(ascending=True)
        if rets:
            s = compound_returns(s, start=1)
            if percent is False:
                percent = True
        mm = s.cummin()
        if percent:
            dup = (s - mm) / mm
        else:
            dup = s - mm
        if maxd:
            return dup.max()
        else:
            return dup
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def semistd(
    s, 
    negative = True, 
    ddof = 1
    ):
    '''
    Computes the semi-volatility of a pd.Dataframe or pd.Series of returns.
    - negative = True, return the semi-volatility of negative return
    - negative = False, return the semi-volatility of positive return
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( semistd, negative=negative, ddof=ddof)
    elif isinstance(s, pd.Series):
        if negative:
            return s[s < 0].std(ddof=ddof) 
        else:
            return s[s >= 0].std(ddof=ddof)
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def annualize_std(
    s, 
    ppy, 
    ddof = 1
    ):
    '''
    Computes the annualized volatility (volatility-per-year) of a pd.Dataframe, pd.Series, or a single returns.
    The variable ppy can be, for example
    12 for weekly, 
    52 for monthly, 
    252 for daily
    data returns. 
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(annualize_std, ppy=ppy)
    elif isinstance(s, pd.Series):
        return s.std(ddof=ddof)*(ppy)**(0.5)
    elif isinstance(s, list):
        return np.std(s, ddof=ddof)*(ppy)**(0.5)
    elif isinstance(s, (int,float)):
        return s * (ppy)**(0.5)
    else:
        raise TypeError("Expected pd.DataFrame, pd.Series, or int/float number")

def var(
    s, 
    CL   = 99/100,
    left = True,
    ):
    '''
    Computes the (1-CL)% Value-at-Risk of a pd.Dataframe or pd.Series of returns.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(var, CL=CL, left=left)
    elif isinstance(s, pd.Series):
        if left:
            return s.quantile(q=1-CL)
        else:
            return s.quantile(q=CL)
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def var_normal(
    s, 
    CL   = 99/100, 
    cf   = False, 
    ddof = 1,
    left = True
    ):
    '''
    Computes the (1-CL)% Value-at-Risk of a pd.Dataframe or pd.Series of returns using the parametric Gaussian method.
    If cf = True, return the Cornish-Fisher cumulants quantile.
    Link: https://www.value-at-risk.net/the-cornish-fisher-expansion/
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(var_normal, CL=CL, cf=cf, ddof=ddof, left=left)
    elif isinstance(s, pd.Series):
        if left: 
            q = stats.norm.ppf(1-CL,loc=0,scale=1)
        else:
            q = stats.norm.ppf(CL,loc=0,scale=1)
        if cf:
            S = s.skew()
            K = kurtosis(s, excess=False)
            q = q + (q**2 - 1)*S/6 + (q**3 - 3*q)*(K-3)/24 - (2*q**3 - 5*q)*(S**2)/36
            #q = q + (q**2 - 1)*S/6 + (q**3 - 3*q)*(K-3*s.std()**2)/24 - (2*q**3 - 5*q)*(S**2)/36
        return s.mean() + q * s.std(ddof=ddof)
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
                             
def es(
    s,        
    CL = 99/100,
    left = True
    ):
    '''
    Computes the (1-CL)% Expected Shortfall of a pd.Dataframe or pd.Series of returns.
    Differently from the 'es' method, the corresponding confidence level scenario is found (no interpolation). 
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(es, CL=CL, left=left)
    elif isinstance(s, pd.Series):
        if left:
            return s[s < var(s, CL=CL, left=True)].mean()
        else:
            return s[s > var(s, CL=CL, left=False)].mean()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def qscenario(s, 
              CL = 99/100):
    '''
    Returns the scenarios of an input series of returns 
    corresponding to the (1-CL)% confidence level.
    The computed scenario is rounded to zero decimal places.
    '''
    wscen = int( round(s.shape[0] * (1 - CL), 0) )
    wscen = 1 if wscen < 1 else wscen
    return wscen

def VaR(s, 
        CL = 99/100,
        left = True
        ):
    '''
    Computes the (1-CL)% Value-at-Risk of a pd.Dataframe or pd.Series of returns.
    Differently from the 'hvar' method, the corresponding confidence level scenario is found (no interpolation). 
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(VaR, CL=CL, left=left)
    elif isinstance(s, pd.Series):
        if left:
            return s.nsmallest(qscenario(s, CL=CL)).values[-1]
        else:
            return s.nsmallest(qscenario(s, CL=1-CL)).values[-1]
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")      
        
def ES(s, 
       CL = 99/100,
       left = True
       ):
    '''
    Computes the (1-CL)% Expected Shortfall of a pd.Dataframe or pd.Series of returns 
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(ES, CL=CL, left=left)
    elif isinstance(s, pd.Series):
        if left:
            return s.nsmallest(qscenario(s, CL=CL)).mean() 
        else:
            return s.nlargest(qscenario(s, CL=CL)).mean()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series") 
        
def summary_stats(
    s, 
    CL     = 99/100, 
    ppy    = 252, 
    excess = False):
    '''
    Returns a dataframe containing annualized returns, annualized volatility, sharpe ratio, 
    skewness, kurtosis, historic VaR, Cornish-Fisher VaR, and Max Drawdown
    '''
    if isinstance(s, pd.Series):
        stats = {
            "(Ann.) Return"       : annualize_returns(s, ppy=ppy),
            "(Ann.) Std"          : annualize_std(s, ppy=ppy),
            "Skewness"            : skewness(s),
            "Kurtosis"            : kurtosis(s, excess=excess),
            f"VaR {CL}"           : var(s, CL=CL),
            f"Normal VaR {CL}"    : var_normal(s, CL=CL, cf=False),
            f"Normal CF VaR {CL}" : var_normal(s, CL=CL, cf=True),
            f"ES {CL}"            : es(s, CL=CL),
            "Max Drawdown"        : drawdown(s, rets=True, maxd=True, percent=True),
            "Minimum"             : s.min(),
            "Maximum"             : s.max()
        }
        return pd.DataFrame(stats, index=[s.name]).T
    
    elif isinstance(s, pd.DataFrame):     
        stats = {
            "(Ann.) Return"       : s.aggregate(annualize_returns, ppy=ppy),
            "(Ann.) Std"          : s.aggregate(annualize_std, ppy=ppy),
            "Skewness"            : s.aggregate(skewness),
            "Kurtosis"            : s.aggregate(kurtosis, excess=excess),
            f"VaR {CL}"           : s.aggregate(var, CL=CL),
            f"Normal VaR {CL}"    : s.aggregate(var_normal, CL=CL, cf=False),
            f"Normal CF VaR {CL}" : s.aggregate(var_normal, CL=CL, cf=True),
            f"ES {CL}"            : s.aggregate(es, CL=CL),
            "Max Drawdown"        : s.aggregate(drawdown, rets=True, maxd=True, percent=True),
            "Minimum"             : s.aggregate(np.min),
            "Maximum"             : s.aggregate(np.max)
        } 
        return pd.DataFrame(stats).T
    

#### Distributions 

def skewness(s):
    '''
    Computes the Skewness of a pd.Dataframe or pd.Series of returns.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(skewness)
    elif isinstance(s, pd.Series):
        # return ( ((s - s.mean()) / s.std(ddof=1))**3 ).mean()
        return (1/s.shape[0])*((s - s.mean())**3).sum() / ((1/(s.shape[0]-1))*((s - s.mean())**2).sum())**(1.5)
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def kurtosis(s, 
             excess = False):
    '''
    Computes the Kurtosis of a pd.Dataframe or pd.Series of returns.
    If excess" is True, returns the "Excess Kurtosis", i.e., Kurtosis minus 3
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(kurtosis, excess=excess)
    elif isinstance(s, pd.Series):
        # return ( ((s - s.mean()) / s.std(ddof=0))**4 ).mean()
        k = ( (1/s.shape[0])*((s - s.mean())**4).sum() ) / ( (1/s.shape[0])*((s - s.mean())**2).sum() )**2
        if excess:
            return k - 3
        else:
            return k
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def is_normal(s,
              siglev = 0.05):
    '''
    Computes the Jarque-Bera test of a pd.Dataframe or pd.Series of returns.
    To see if a series (of returns) is normally distributed.
    Returns True or False according to whether the p-value 
    is larger than input significance level=0.01.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(is_normal, siglev=siglev)
    elif isinstance(s, pd.Series):
        statistic, pvalue = stats.jarque_bera(s)
        return pvalue > siglev
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")
                
def dist_normal(
    x, 
    mu  = 0, 
    std = 1,
    cum = False
    ):
    '''
    Normal (or Gaussian) distribution:
    - cum   : False for evaluating the PDF at input point x
            : True for evaluating the CDF at input point x
    Using scipy.stats.
    '''
    if cum:
        return stats.norm.cdf(x, loc=mu, scale=std)
    else:
        return stats.norm.pdf(x, loc=mu, scale=std)

def gen_pdf_normal(
    a   = -5, 
    b   = 5, 
    mu  = 0, 
    std = 1, 
    dx  = 0.01,
    ):
    '''
    Generation of the Normal (or Gaussian) Distribution PDF with mean 'mu' and 
    variance 'std', within the range (a,b).
    Returns a pd.Series.
    '''
    xx = np.arange(a, b, dx)
    pdf = [dist_normal(x, mu=mu, std=std, cum=False) for x in xx]
    pdf = pd.Series(pdf, index=xx)
    pdf.name = "Normal"
    return pdf

def gen_cdf_normal(
    a   = -5, 
    b   = 5, 
    mu  = 0, 
    std = 1, 
    dx  = 0.01,
    ):
    '''
    Generation of the Normal (or Gaussian) Distribution CDF with mean 'mu' and 
    variance 'std', within the range (a,b).
    Returns a pd.Series.
    '''
    xx = np.arange(a, b, dx)
    cdf = [dist_normal(x, mu=mu, std=std, cum=True) for x in xx]
    cdf = pd.Series(cdf, index=xx)
    cdf.name = "Normal"
    return cdf

def gen_rvs_normal(
    mu   = 0,
    std  = 1,
    size = 1000
    ):
    '''
    Returns a pd.Series of random variables normally distributed.
    Using scipy.stats.
    '''
    return pd.Series(stats.norm.rvs(loc=mu, scale=std, size=size))

def Gamma(x):
    '''
    Returns the Gamma function (from math)
    '''
    return gamma(x) if x != 0 else 1

def dist_tstudent(
    x, 
    df    = 3,
    mu    = 0,
    scale = 1, 
    stdz  = False,
    cum   = False 
    ):
    '''
    t-Student Distribution.
    Note that t-Student distribution variance exists for df > 2.
    - stdz  : True for evaluating the "Standardized" t-Student.
              It has mean = mu and variance = 1, if df>2
            : False for evaluating the "non-Standardized" (or "Standard") t-Student 
              It has mean = mu and variance = scale**2 * df/(df-2), if df>2
    '''
    if cum:
        if stdz:
            return stats.t.cdf(x, df=df, loc=mu, scale=((df-2)/df)**0.5)
        else:
            return stats.t.cdf(x, df=df, loc=mu, scale=scale)
    else:
        if stdz:
            return stats.t.pdf(x, df=df, loc=mu, scale=((df-2)/df)**0.5)
        else:
            return stats.t.pdf(x, df=df, loc=mu, scale=scale) 

def gen_pdf_tstudent(
    a     = -5, 
    b     = 5, 
    df    = 3,
    mu    = 0, 
    scale = 1, 
    stdz  = False, 
    dx    = 0.01,
    ):
    '''
    Generation of the t-Student Distribution PDF within the range(a,b).
    Note that t-Student distribution variance exists for df > 2.
    - stdz  : True for evaluating the "Standardized" t-Student.
              It has mean = mu and variance = 1, if df>2
            : False for evaluating the "non-Standardized" (or "Standard") t-Student 
              It has mean = mu and variance = scale**2 * df/(df-2), if df>2
    '''
    xx = np.arange(a, b, dx)
    pdf = [dist_tstudent(x, df=df, mu=mu, scale=scale, stdz=stdz, cum=False) for x in xx]
    pdf = pd.Series(pdf, index=xx)
    if stdz:
        pdf.name = "Standarized t"
    else:
        pdf.name = "Non-Standardized t"
    return pdf

def gen_cdf_tstudent(
    a     = -5, 
    b     = 5, 
    df    = 3,
    mu    = 0, 
    scale = 1, 
    stdz  = False, 
    dx    = 0.01,
    ):
    '''
    Generation of the t-Student Distribution CDF within the range(a,b).
    Note that t-Student distribution variance exists for df > 2.
    - stdz  : True for evaluating the "Standardized" t-Student.
              It has mean = mu and variance = 1, if df>2
            : False for evaluating the "non-Standardized" (or "Standard") t-Student 
              It has mean = mu and variance = scale**2 * df/(df-2), if df>2
    '''
    xx = np.arange(a, b, dx)
    cdf = [dist_tstudent(x, df=df, mu=mu, scale=scale, stdz=stdz, cum=True) for x in xx]
    cdf = pd.Series(cdf, index=xx)
    if stdz:
        cdf.name = "Standarized t"
    else:
        cdf.name = "Non-Standardized t"
    return cdf

def gen_rvs_tstudent(
    df    = 3, 
    mu    = 0,
    scale = 1,
    size  = 1000,
    stdz  = False,
    ):
    '''
    Returns a pd.Series of random variables t-Student distributed.
    t-Student distribution variance exists for df > 2.
    - stdz  : True for the "Standardized" t-Student.
              It has mean = mu and variance = 1, if df>2
            : False for the "non-Standardized" (or "Standard") t-Student 
              It has mean = mu and variance = scale**2 * df/(df-2), if df>2
    '''
    if stdz:
        return pd.Series(stats.t.rvs(df=df, loc=mu, scale=((df-2)/df)**0.5, size=size))
    else:
        return pd.Series(stats.t.rvs(df=df, loc=mu, scale=scale, size=size))

def some_pdf(x, mu, std): 
    '''
    Returns a user-defined pdf for (general) generation of random variables.
    To be used with "gen_rvs_from_pdf" method.
    The example returns the normal pdf.
    '''
    pdf = 1/np.sqrt(2*np.pi*std**2)*np.exp(-0.5*((x - mu)/std)**2)
    return pdf

def gen_rvs_from_pdf(
    pdf, 
    size   = 1000,
    iguess = 0.0,
    tol    = 1e-3, 
    **kwargs
    ):   
    '''
    Generates random variables distributed according to the input (user-defined) pdf function.
    Input "pdf" should be a function and "kwargs" would be extra pdf input parameters.
    For example, if "some_pdf" returns the normal distribution, then:
    -> gen_rvs_from_pdf(some_pdf, mu=0, std=1)
    returns normally distributed random variables with mean mu=0 and std=1.
    ''' 
    def objective(x, pdf, *args):
        return abs( (integrate.quad(pdf, -1e2, x, args=(args[0]))[0] - pu) )

    args = tuple()
    for key in kwargs.keys():
        args = args + (kwargs[key],)

    # Slow...
    rup = np.random.uniform(0,1,size)
    rvs = []
    for pu in rup:
        result = minimize(objective, 
                    iguess,
                    args    = (pdf, args),
                    method  = "SLSQP",
                    options = {"disp": False},
                    tol     = tol,
                    bounds  = None
                    )
        rvs.append(result.x[0])
    return pd.Series(rvs)

def distfit(s, 
            dtype: str  = "t",
            pdf: bool   = False,
            mm: float   = 0,
            dx: float   = 0.05) -> dict:
    '''
    Best-Fit distribution approximation using Maximum-Likelihood-Estimation via scipy.stats. 
    Returns a dictionary with distribution parameters, e.g., mean and standard location.
    - dist  : "n" for Normal distribution fit 
            : "t" for t-Student distribution fit
            : "gdp" for Generalized Pareto distribution fit
    - pdf   : if True, returns a vector with the fitted pdf
    '''
    if pdf:
        x = np.arange(s.min()-mm, s.max()+mm, dx)
    
    if dtype == "n":
        # Normal fit
        mu, std = stats.norm.fit(s)
        if pdf:
            npdf = stats.norm.pdf(x, mu, std)
            return dict({'mu': mu, 'std': std, 'pdf': npdf, 'x': x, 'dx': dx})
        else:
            return dict({'mu': mu, 'std': std})
    
    elif dtype == "t":
        # t-Student fit
        df, mu, scale = stats.t.fit(s)
        if pdf:
            tpdf = stats.t.pdf(x, df, mu, scale)
            return dict({'df': df, 'mu': mu, 'scale': scale, 'pdf': tpdf, 'x': x, 'dx': dx})
        else:
            return dict({'df': df, 'mu': mu, 'scale': scale})
    
    elif dtype == "gdp":
        # Generalized Pareto fit:
        # pdf = (1 + c*x)^(-1-1/c)
        # con c = shape parameter
        c, mu, std = stats.genpareto.fit(s)
        # Note that 
        # pdf = stats.genpareto.pdf(x, c, mu, std)
        # is equivalent to standardize the pdf by using mu and std 
        if pdf:
            pdf = (1/std)*(1 + c/std*(x-mu))**(-1-1/c)
            return dict({'c': c, 'mu': mu, 'std': std, 'pdf': pdf, 'x': x, 'dx': dx})
        else:
            return dict({'c': c, 'mu': mu, 'std': std})
        
    else:
        raise ValueError("Enter valid distribution value")
    

def empirical_cdf(s) -> pd.Series:
    '''
    Returns the Empirical Cumulative Distribution Function (ECDF)
    of an input return series
    '''
    F_ecdf = ECDF(s)    
    F_emp = pd.Series(F_ecdf.y, index=F_ecdf.x, name="ECDF")
    F_emp = F_emp.drop(index=F_emp.index[0])
    return F_emp


def hypothetical_cdf(s, 
                    #  dist: dict, 
                     dtype: str = "t") -> pd.Series:
    
    dist = distfit(s, dtype=dtype, pdf=False)

    if dtype == "n":
        cdf = stats.norm.cdf(s, loc=dist["mu"], scale=dist["std"])
        name = "Fitted Normal CDF"
    
    if dtype == "t":
        cdf = stats.t.cdf(s, df=dist["df"], loc=dist["mu"], scale=dist["scale"])  
        name = "Fitted t-Student CDF"

    return pd.Series(np.sort(cdf), index=np.sort(s), name=name)


#### Covariances and Correlations

def sample_cov(
    r, 
    ddof = 1
    ):
    '''
    Returns the sample covariance matrix (n-by-n) of a pd.DataFrame of n times series returns 
    '''
    if isinstance(r, pd.Series):
        return r.std(ddof=ddof)**2
    elif isinstance(r, pd.DataFrame):
        return r.cov(ddof=ddof)
    else:
        raise ValueError("Expected df to be a pd.Series or pd.DataFrame of returns")


def cc_cov(r):
    '''
    Returns a covariance matrix using the Elton/Gruber Constant Correlation model
    '''
    # Correlation matrix  
    rhos = r.corr()
    # Mean Correlation: since the matrix rhos is a symmetric with diagonals all 1,
    # the mean correlation can be computed by:
    mean_rho = (rhos.values.sum() - rhos.shape[0]) / (rhos.shape[0]**2 - rhos.shape[0]) 
    # Constant correlation matrix containing 1 on the diagonal and the mean correlation outside
    ccor = np.full_like(rhos, mean_rho)
    np.fill_diagonal(ccor, 1.0)
    # New covariance matrix by multiplying mean_rho * std_i**2, 
    # the product of the stds is done via np.outer
    ccov = ccor * np.outer(r.std(), r.std())
    return pd.DataFrame(ccov, index=r.columns, columns=r.columns)


def shrinkage_cov(r, delta=0.5):
    '''
    Returns a convariance matrix computed via Statistical Shrinkage method
    which 'shrinks' between the constant correlation and standard sample covariance estimators 
    '''
    return delta*cc_cov(r) + (1-delta)*sample_cov(r)