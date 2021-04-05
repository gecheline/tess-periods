import numpy as np 
import pandas as pd
import * from ../lc_geometry.py 
import * from ../mcmc_validated.py

results = pd.read_csv("data/tess_refined_2g.csv")
results = results.replace([np.inf, -np.inf], np.nan)


for i in range(len(results.values)):
    #C,mu1,d1,sigma1,mu2,d2,sigma2,Aell
    print(i)
    if np.isnan(results['logprob_mean_2g'][i]):
        pass 

    tic = results['TIC'][i]
    filename = 'data/lcs_ascii/tic'+f'{tic:010}'+'.norm.lc'
    lc = np.loadtxt(filename)

    period = results['period_2g'][i]
    t0 = results['t0_2g'][i]
    model_params = np.array([results['C'][i],
                                results['mu1'][i],
                                results['d1'][i],
                                results['sigma1'][i]
                                results['mu2'][i]
                                results['d2'][i]
                                results['sigma2'][i]
                                results['Aell'][i]])
    model_params = model_params[~np.isnan(model_params)]


    phases, fluxes_ph, sigmas_ph = fold_with_period(lc[:,0], lc[:,1], lc[:,2], period=period, t0=t0)
    # compute model with the selected twog function
    func = results['model_2g'][i]
    model = twogfuncs[func](phases, *model_params)

    np.savetxt('data/lcs_2g/tic'+f'{tic:010}'+'.2g.lc', 
                np.array([phases, model]).T))
