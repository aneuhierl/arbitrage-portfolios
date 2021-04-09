import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
from scipy.sparse.linalg import eigsh

pandas2ri.activate()
########################################################################
# Parameters
########################################################################
n_eigenvectors   = 6
window_len_proj  = 12 # months
target_vol       = 0.2 # annualized

########################################################################
# Data
########################################################################
X = robjects.r['readRDS']('kkn-replication-data.RDS')
X['date'] = pd.to_datetime(X['date'], unit='D')

# column names of the characteristics
characteristics = X.columns[4:].to_list()

########################################################################
# actual program here
########################################################################
nT = X['t'].max() - 1
est_t = range(window_len_proj, X['t'].max(), 1)
weight_l = []
for t in est_t:
    # some output
    if (t - window_len_proj + 1) % 20 == 0:
        print('Currently on estimation ', (t - window_len_proj + 1), 'of ', (nT - window_len_proj + 1), '-- #Factors: ', n_eigenvectors)

    # this creates a balanced panel for the estimation period and also the prediction period
    ind = list(i for i in range(t - window_len_proj + 1, t + 2, 1))
    Y = X.loc[X['t'].isin(ind)].copy()
    Y['keep'] = Y[['permno', 't']].groupby(['permno']).transform(lambda x: sorted(list(set(x))) == ind)
    Y = Y[Y.keep == True]
    Y = Y.drop(columns='keep').reset_index(drop=True)

    # rank transform characteristics to [0,1]
    Y[characteristics] = Y.groupby(['t']).transform(lambda x: (1/(len(x) + 1)) * (np.argsort(np.argsort(np.array(x), kind='mergesort'), kind='mergesort') + 1))[characteristics]

    # subsetting for convenience
    YA = Y.loc[Y['t'] == ind[len(ind) - 1]] # prediction period
    Y = Y.loc[Y['t'] < ind[len(ind) - 1]] # restrict the estimation period to the time before the prediction period

    # de-mean returns and project de-meaned returns on the characteristics
    Y['ret_demean'] = Y[['permno', 'ret']].groupby(['permno']).transform(lambda x: x - np.mean(x))

    # projection step
    r = Y['ret_demean'].to_numpy()
    Z = Y.loc[:, characteristics].to_numpy()
    Z = Z - np.mean(Z, axis=0)
    B = np.linalg.solve(np.dot(Z.transpose(), Z), np.dot(Z.transpose(), r))
    yhat = np.dot(Z, B)
    Y['rhat'] = yhat

    # reshape return projection and return
    R = Y.pivot_table(index=['permno'], columns='date', values='ret').reset_index().drop(columns=['permno']).to_numpy()
    Rhat = Y.pivot_table(index=['permno'], columns='date', values='rhat').reset_index().drop(columns=['permno']).to_numpy()

    # Eigendecomposition
    RR = np.dot(Rhat, Rhat.transpose())
    ED = eigsh(RR, k=n_eigenvectors, which='LM')
    GB = np.flip(ED[1], axis=1)

    # solve constrained LS problem
    RB = R.mean(axis=1)
    ZP = Y.loc[Y['t'] == ind[0], ['date', 'permno', 'ret', 'ret_demean'] + characteristics]
    Z = ZP.loc[:, characteristics].to_numpy()
    Z = Z - np.mean(Z)
    E = Z - np.dot(np.dot(GB, GB.transpose()), Z)
    theta = np.linalg.solve(np.dot(E.transpose(), E), np.dot(E.transpose(), RB))

    # update characteristics
    Z = YA[characteristics].to_numpy()

    # normalize
    Z = Z - np.mean(Z)

    # G_X_Alpha
    E = Z - np.dot(np.dot(GB, GB.transpose()).transpose(), Z)
    GXA = np.dot(E, theta)

    # scale towards a target vol
    R_alpha = np.dot(GXA.transpose(), R)
    sd_alpha = np.std(R_alpha, ddof=1)
    sd_scale_factor = (target_vol / np.sqrt(12) / sd_alpha)
    GXA = GXA * sd_scale_factor

    #########################################################
    # OUTPUT
    #########################################################
    oos_date = YA['date'].iloc[0]
    weight_l.append(pd.DataFrame({'permno': YA['permno'], 'date': oos_date, 'weight': GXA}))

# bind
weight_dt = pd.concat(weight_l, axis = 0)

# merge portfolio weights with returns and compute portfolio returns and Sharpe
R_alpha = pd.merge(left=X[['permno', 'date', 'ret']], right=weight_dt, how='inner', on=['permno', 'date'], sort=True)
R_alpha_out = R_alpha.groupby(['date'], as_index=True, sort=True).apply(lambda x: pd.Series({'r_alpha':sum(x.weight * x.ret)})).reset_index()
R_alpha_out = R_alpha_out.loc[R_alpha_out['date'].dt.year >= 1968]
print(pd.DataFrame({'ann_mean':12 * np.mean(R_alpha_out['r_alpha']), 'ann_sd':np.sqrt(12) * np.std(R_alpha_out['r_alpha'], ddof=1), 'sharpe':np.sqrt(12) * np.mean(R_alpha_out['r_alpha']) / np.std(R_alpha_out['r_alpha'], ddof=1)}, index=[0]))