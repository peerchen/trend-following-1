import os
import itertools
import pickle
import numpy
import pandas
from pandas import Series, DataFrame
from datetime import datetime
import matplotlib
from matplotlib import pyplot
import matplotlib.transforms
import quandl

from scipy.signal import gaussian, find_peaks, find_peaks_cwt
from scipy.ndimage import filters


    
# Utils
def dict_list_to_DataFrame(dl): 
    dl = [{k: [i] for k, i in d.items()} for d in dl]
    out = pandas.concat([DataFrame.from_dict(d) for d in dl], sort=False)
    return out

def plot_prices(prices, name=''):
    fig, ax = pyplot.subplots()
    ax.plot(prices.Open)
    ax.plot(prices.High)
    ax.plot(prices.Low)
    ax.plot(prices.Close)
    ax.axhline(0, color='grey', lw=2, alpha=0.75)
    ax.set_title(name)
    ax.legend()


def daily_to_weekly_prices(d_prices):
    return d_prices.resample(rule='W-FRI').apply({'Open': 'first',
                                                  'Close': 'last',
                                                  'High': 'max',
                                                  'Low': 'min',
                                                  'Volume': 'sum'})


# Data loading

QUANDL_PATH = 'input/Quandl/'

# Free sample tickers from Exchange Data International
QUANDL_FREE_SAMPLES_EDI = {
    # https://www.quandl.com/data/XNAS-NASDAQ-Stock-Market-Prices
    'XNAS': ['AAL', 'AAME', 'AAON', 'AAPL', 'AAXJ', 'ABAC', 'ABAX',
             'ABCO', 'ABG', 'ABM', 'ABTL', 'ACET', 'ACIW', 'ACLS', 'ACNB'],
    # https://www.quandl.com/data/XBOM-Bombay-Stock-Exchange-Prices
    'XBOM': ['500002', '500003', '500008', '500010', '500012',
             '500013', '500020', '500023', '500024', '500027',
             '500031', '500032', '500038', '500039', '500040'],
    # https://www.quandl.com/data/XTSE-Toronto-Stock-Exchange-Prices
    'XTSE': ['AAB', 'ABT', 'ABX', 'ACD', 'ACQ', 'AEM', 'AFN', 'AGT',
             'AGU', 'AIF', 'ALA', 'ALB', 'ALC', 'ALO', 'AND'],
    # https://www.quandl.com/data/XSHG-Shanghai-Stock-Exchange-Prices
    'XSHG' : ['600000', '600004', '600006', '600007', '600009',
              '600010', '600011', '600012', '600015', '600016',
              '600017', '600019', '600020', '600021', '600026'],
    # https://www.quandl.com/data/XLON-London-Stock-Exchange-Prices
    'XLON': ['AAIF', 'AAL', 'AAS', 'ABBY', 'ABC', 'ABF', 'ADIG', 
             'ADM', 'ADT', 'AEFS', 'AEO', 'AEP', 'AFN', 'AFS', 'AGK'],
    # https://www.quandl.com/data/XSES-Singapore-Exchange-Prices
    'XSES': ['1B6', '1C0', 'A04', 'A05', 'AFC', 'AGS', 'AUE', 'AVX', 
             'BBW', 'BCD', 'BCV', 'BCX',  'BCY', 'BEC', 'BESU'],
    # https://www.quandl.com/data/XNYS-New-York-Stock-Exchange-Prices
    'XNYS': ['A', 'AAT', 'AB', 'ABB', 'ABBV', 'ABC', 'ABEV', 'ABG', 
             'ABM', 'ABR', 'ABT', 'ABX', 'ACC', 'ADC', 'ADM'],
    # https://www.quandl.com/data/XHKG-Hong-Kong-Stock-Exchange-Prices
    'XHKG': ['00002', '00003', '00004', '00005', '00006',
             '00008', '00010', '00011', '00012', '00014',
             '00015', '00016', '00017', '00018', '00019'],
    # https://www.quandl.com/data/XASE-NYSE-MKT-AMEX-Prices
    'XASE': ['ABE', 'ACU', 'ACY', 'ADK', 'AE',  'AMS', 'ARNC_P',
             'BAA', 'BDL', 'BFY', 'BHB', 'BHV', 'BLE', 'BLJ', 'BTI'],
    # https://www.quandl.com/data/XNSE-National-Stock-Exchange-of-India-Prices
    'XNSE': ['ABB', 'ACC', 'ACE', 'ADSL', 'AFL', 'ALICON',
             'BAJAJ_AUTO', 'BASF', 'BASML', 'BBL', 'BEL',
             'BEPL, BHEL', 'BIL', 'BIOCON'],
    # https://www.quandl.com/data/XTSX-Toronto-Ventures-Stock-Exchange-Prices
    'XTSX': ['ABI', 'ABN', 'ADD', 'ADK', 'ADL', 'AFCC', 'AFM', 'AGD', 
             'AGO', 'AHR', 'AIIM', 'ALT', 'ALZ', 'AME', 'AMK'],
    # https://www.quandl.com/data/XSHE-Shenzhen-Stock-Exchange-Prices
    'XSHE': ['200011', '200012', '200018', '200025', '200026',
             '200055', '200056', '200413', '200418', '200488',
             '200521', '200530', '200539', '200541', '200550'],
    # https://www.quandl.com/data/XJPX-Japan-Exchange-Group-Prices/documentation/coverage
    'XJPX': ['13010', '13050', '13060', '13080', '13100', '13110', 
             '13200', '13290', '13300', '13320', '13430', '13440', 
             '13450', '13480', '13760']
}

# xjpx_df = DataFrame(data = numpy.arange(len(QUANDL_FREE_SAMPLES_EDI['XJPX'])),
#                     index=['XJPX/' + i for i in QUANDL_FREE_SAMPLES_EDI['XJPX']])
# xjpx_df.to_csv(path_or_buf='Input/Quandl/XJPX.csv', header=False)


def saf_quandl_get(dataset, **kwargs):
    try:
        return quandl.get(dataset, **kwargs)
    except:
        return None


def get_quandl_edi(exchanges = 'XNAS',
                   free=True, download=False,return_df=True,
                   verbose=False):
    """
    Downloads price series from Quandl vendor Exchange Data International
    
    Parameters
    ----------
    exchanges : List with names of the exchanges from which to download prices.
    free : If True, only free sample prices are downloaded.
    download : 
        If True, downloads the prices from quandl.  
        If False, looks for previously downloaded results in the QUANDL_PATH folder.
    verbose : If True, prints downloaded tickers.
    
    Returns
    -------
    out : a dict of pandas DataFrame for each ticker.
    """
    
    out = dict()
    
    if download:
        for x in exchanges:
            
            prices = pandas.read_csv(QUANDL_PATH + 'EDI/' + x + '.csv', names=['Ticker', 'Desc.'])
            free_sample = QUANDL_FREE_SAMPLES_EDI[x]
            which_free = [re.search('|'.join(free_sample), t) is not None and
                          re.search('_UADJ', t) is None
                          for t in prices['Ticker']]
            if free: 
                prices = prices[which_free]
                
            if verbose:
                print('Downloading prices from', x, '...')
                
            out[x] = {t: saf_quandl_get(t) for t in prices['Ticker']}
            out[x] = {k: i for k, i in out[x].items() if i is not None}
            
            with open(QUANDL_PATH + 'EDI/' + x + '.pickle', 'wb') as f:
                pickle.dump(out[x], f, pickle.HIGHEST_PROTOCOL)
            
            if verbose:
                print(list(out[x].keys()))
    
    else:
        for x in exchanges:
            try:
                with open(QUANDL_PATH + 'EDI/' + x + '.pickle', 'rb') as f:
                    out[x] = pickle.load(f)
            except:
                pass
    
    
    out = {k: i for x in out.keys() for k, i in out[x].items()}
    out = {k: i[['Open', 'High', 'Low', 'Close', 'Volume']] for k, i in out.items()}
    
    if return_df:
        
        def add_ticker(price, ticker):
            price['Ticker'] = ticker
            return price.reset_index().set_index(['Ticker', 'Date'])
        
        tickers = list(out.keys())
        out = pandas.concat([add_ticker(out[t], t) for t in tickers])
        
        return tickers, out
    else:
        return out


def get_quandl_sharadar(free=True, download=False):
    """
    Downloads price series from Quandl dataset Sharadar Equity Prices
    
    Parameters
    ----------
    free : If True, only free sample prices are downloaded.
    download : 
        If True, downloads the prices from quandl.  
        If False, looks for previously downloaded results in the QUANDL_PATH folder.
    
    Returns
    -------
    out : a dict of pandas DataFrame for each ticker.
    """
    
    if free:
        if download:
            sharadar = quandl.get_table('SHARADAR/SEP', paginate=True)
            sharadar = sharadar.rename({n: n.title() for n in sharadar.keys().values}, axis=1)
            sharadar = sharadar.reset_index().drop('None', axis=1)
            sharadar.to_feather(fname=QUANDL_PATH + 'Sharadar/sharadar_free.feather')
        else:
            sharadar = pandas.read_feather(path=QUANDL_PATH + 'Sharadar/sharadar_free.feather')
            
    else:
        if download:
            sharadar = pandas.read_csv(filepath_or_buffer='input/Quandl/Sharadar/sharadar_full.csv')
            sharadar = sharadar.rename({n: n.title() for n in sharadar.keys().values}, axis=1)
            sharadar.to_feather(fname=QUANDL_PATH + 'Sharadar/sharadar_full.feather')
        else:
            sharadar = pandas.read_feather(path=QUANDL_PATH + 'Sharadar/sharadar_full.feather')
    
    tickers = list(set(sharadar.Ticker))
    sharadar.Date = pandas.to_datetime(sharadar.Date)
    sharadar = sharadar.set_index(['Ticker', 'Date'])
    
    return tickers, sharadar



def clean_sharadar(prices):
    """
    Assets to check: NXG
    """
    prices = prices.query('Volume > 0')
    
    return prices



def get_sharadar_train():
    
    prices = pandas.read_feather(QUANDL_PATH + 'Sharadar/sharadar_train.feather').set_index(['Ticker', 'Date'])
    dir_train = os.listdir(QUANDL_PATH + 'Sharadar/train/')
    tickers = [f.replace('.feather', '') for f in dir_train]

    return tickers, prices


def get_sharadar_dev():
    
    prices = pandas.read_feather(QUANDL_PATH + 'Sharadar/sharadar_dev.feather').set_index(['Ticker', 'Date'])
    dir_dev = os.listdir(QUANDL_PATH + 'Sharadar/dev/')
    tickers = [f.replace('.feather', '') for f in dir_dev]

    return tickers, prices


def get_sharadar_test():
    
    prices = pandas.read_feather(QUANDL_PATH + 'Sharadar/sharadar_test.feather').set_index(['Ticker', 'Date'])
    dir_test = os.listdir(QUANDL_PATH + 'Sharadar/test/')
    tickers = [f.replace('.feather', '') for f in dir_test]

    return tickers, prices


## Preparing the data for machine learning...

def smooth_price(df, sd=20., N=10000, double=False):
    """
    Applies a gaussian filter to the closing price in ohlc data frame.
    """
    N = max(N, 4 * sd)
    f_ga = gaussian(N, std=sd)
    f_ga = f_ga / f_ga.sum()
    if double:
        df = df.assign(Smoothed=filters.convolve1d(filters.convolve1d(df.Close, f_ga), f_ga))
    else:
        df = df.assign(Smoothed=filters.convolve1d(df.Close, f_ga))
        
    return df


def find_trends(df, sd=20., N=10000, double=False):
    """
    Finds the trends and the maximum drawdown within trends for a Close price series.
    """
    # Peaks and valleys of shoothed series
    df = smooth_price(df, sd, N, double)
    df = df.assign(Trend=numpy.nan, n_Trend=numpy.nan, Max_Drawdown=numpy.nan)
    
    peaks, _ = find_peaks(df.Smoothed)
    valleys, _ = find_peaks(-df.Smoothed)

    n_changes = min(len(peaks), len(valleys))
    if n_changes == 0:
        if df.Smoothed[-1] > df.Smoothed[0]:
            peaks = numpy.ones(1, dtype=numpy.int32) * len(df) - 1
        else:
            valleys = numpy.ones(1, dtype=numpy.int32) * len(df) - 1
    else:
        if valleys.max() > peaks.max(): # Last
            peaks = numpy.concatenate((peaks, numpy.ones(1, dtype=numpy.int32) * len(df) - 1))
        else:
            valleys = numpy.concatenate((valleys, numpy.ones(1, dtype=numpy.int32) * len(df) - 1))
        
    

    df.loc[df.index[peaks], 'Trend'] = 1
    df.loc[df.index[valleys], 'Trend'] = -1
    df.Trend.fillna(method='bfill', inplace=True)

    
    # Max drawdown of long position when trending up, short position when trending down.
    breakpoints = numpy.concatenate((numpy.zeros(1, dtype=numpy.int32), peaks + 1, valleys + 1))
    breakpoints.sort()
    
    for b in range(1, len(breakpoints)):
        trend_start = breakpoints[b - 1]
        trend_end = breakpoints[b]
        res_b = df[trend_start:trend_end]
        trend_b = res_b.Trend[0]

        # Adjust for position (long, short)
        close_b = res_b.Close[0] + trend_b * (res_b.Close - res_b.Close[0])
        close_b = close_b.sort_index()
        
        # True range
        true_range_b = (numpy.max((res_b.High, res_b.Close.shift().fillna(method='bfill')), axis=0) - \
                        numpy.min((res_b.Low, res_b.Close.shift().fillna(method='bfill')), axis=0)) / res_b.Close
        
        ratio = close_b[-1] / close_b[0]
        if len(close_b) > 1:
            icagr = numpy.log(ratio) * (364.25 / (close_b.index[-1] - close_b.index[0]).components.days)
        else:
            icagr = numpy.zeros(1, dtype=numpy.float64)

        peak = close_b[0]
        low = peak
        drawdown = 0
        max_drawdown = 0
        for i in range(1, len(close_b)):
            # Max drawdown
            if close_b[i] > peak:
                peak = close_b[i]
                low = peak
            if close_b[i] < low:
                low = close_b[i]
                drawdown = low / peak - 1
            max_drawdown = min(drawdown, max_drawdown)
            
        if max_drawdown != 0:
            bliss = - icagr / max_drawdown
        else:
            bliss = numpy.nan

        df.loc[res_b.index, 'n_Trend'] = int(b)
        df.loc[res_b.index[0], 'Max_Drawdown'] = - max_drawdown
        df.loc[res_b.index[0], 'ATR'] = true_range_b.mean()
        df.loc[res_b.index[0], 'max_TR'] = true_range_b.max()
        df.loc[res_b.index[0], 'min_TR'] = true_range_b.min()
        df.loc[res_b.index[0], 'Ratio'] = ratio
        df.loc[res_b.index[0], 'ICAGR'] = icagr
        df.loc[res_b.index[0], 'Bliss'] = bliss
        df.loc[res_b.index, 'Trend_Start'] = res_b.index[0]
        df.loc[res_b.index, 'Trend_End'] = res_b.index[-1]
    
    return df
    
    
def summarise_trends(df, sd=20., N=10000):
    trends = find_trends(df, sd, N)
    total_ratio = trends.groupby('n_Trend').first().Ratio.product()
    total_icagr = numpy.log(total_ratio) * (364.25 / (trends.index[-1] - trends.index[0]).components.days)
    mean_icagr = trends.groupby('n_Trend').first().ICAGR.mean()
    neg_icagr = numpy.sum(trends.groupby('n_Trend').first().ICAGR < 0)
    mean_bliss = trends.groupby('n_Trend').first().Bliss.dropna().mean()
    max_drawdown = trends.groupby('n_Trend').first().Max_Drawdown.max()
    if max_drawdown > 0:
        bliss = total_icagr / max_drawdown
    else:
        bliss = numpy.nan
    
    neg_freq = neg_icagr.astype(numpy.float64) / trends.n_Trend.max().astype(numpy.float64)
    
    res = DataFrame(trends.groupby('n_Trend').Trend.count().describe())
    res = res.transpose().assign(sd=sd, n_days=len(df)).reset_index().drop('index', axis=1)
    res = res.assign(trend_freq=364.25*res['count'].astype(numpy.float64)/res.n_days.astype(numpy.float64))
    res = res.assign(Ratio=total_ratio,
                     ICAGR=total_icagr, mean_ICAGR=mean_icagr,
                     neg_ICAGR=neg_icagr, neg_freq=neg_freq,
                     Bliss=bliss, mean_Bliss=mean_bliss, Max_Drawdown=max_drawdown)
    
    return res


def plot_trends(df, tit=''):
    pal = pyplot.get_cmap('Paired').colors
    
    fig, ax = pyplot.subplots(figsize=(16, 5))
    trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    if len(set(df.Trend.values)) > 1:
        ax.fill_between(df.index, 0, df.Trend.max(), where=df.Trend > 0, facecolor=pal[0],
                        alpha=0.25, transform=trans, label='Trend up')
        ax.fill_between(df.index, 0, df.Trend.max(), where=df.Trend < 0, facecolor=pal[4],
                        alpha=0.25, transform=trans, label='Trend down')
    pyplot.plot(df.Close, label='Close')
    pyplot.plot(df.Smoothed, label='Smoothed')
    pyplot.plot(df.Close * (1 - df.Max_Drawdown * df.Trend), label='Stop-loss', alpha = 0.5)
    pyplot.axhline(0, c='grey')
    pyplot.legend()
    pyplot.title(tit)
    pyplot.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


