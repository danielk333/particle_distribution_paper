import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

_font = 22
plt.style.use('dark_background')
plt.rc('text', usetex=True)

mu_target, sig_target = 0.0, 1.0
mu_source, sig_source = 2.0, 0.1
integ_min, integ_max = 1.8, 2.2
samples = 100

Pf_analytic = st.norm.cdf(integ_max, mu_target, sig_target) - st.norm.cdf(integ_min, mu_target, sig_target)


x_vec = np.linspace(-2, 3, num=1000)

fig, ax = plt.subplots(1, 1, figsize=(12,8))

ax.plot(x_vec, st.norm.pdf(x_vec, mu_source, sig_source), 'r', label='Sampling distribution')
ax.plot(x_vec, st.norm.pdf(x_vec, mu_target, sig_target), 'b', label='Target distribution')
ax.axvline(integ_min, color='g', label='Integration region')
ax.axvline(integ_max, color='g')
ax.set_xlabel('Some important variable', fontsize=_font)
ax.set_ylabel('Probability', fontsize=_font)
ax.set_title('Target versus Sampling distribution', fontsize=_font + 6)
ax.tick_params('both', labelsize=_font-4)
ax.legend(fontsize=_font)


def sample():
    #do importance sampling
    x = np.random.randn(samples)*sig_source + mu_source

    pi = lambda x: st.norm.pdf(x, mu_source, sig_source)
    f = lambda x: st.norm.pdf(x, mu_target, sig_target)

    x_a = x[x >= integ_min]
    x_a = x_a[x_a <= integ_max]

    Ppi = float(x_a.size)/float(samples)

    #importance sampling formula
    h = f(x_a)/pi(x_a)
    Pf = np.mean(h)*Ppi

    #do a direct monte carlo estimation as comparison
    x_dmc = np.random.randn(samples)*sig_target + mu_target
    x_dmc_a = x_dmc[x_dmc >= integ_min]
    x_dmc_a = x_dmc_a[x_dmc_a <= integ_max]
    Pf_dmc = float(x_dmc_a.size)/float(samples)

    return Pf, Pf_dmc


Pf, Pf_dmc = sample()

print(f'True value     : {Pf_analytic}')
print('-'*20)
print(f'Estimated value: {Pf}')
print(f'Error          : {Pf_analytic - Pf}')
print('-'*20)
print(f'Direct value   : {Pf_dmc}')
print(f'Error          : {Pf_analytic - Pf_dmc}')
print('-'*20)


est_n = 3000
#lets get the error distribution
IS_err = np.empty((est_n, ))
DMC_err = np.empty((est_n, ))

for ind in range(est_n):

    Pf, Pf_dmc = sample()

    IS_err[ind] = Pf_analytic - Pf
    DMC_err[ind] = Pf_analytic - Pf_dmc

fig, ax = plt.subplots(1, 1, figsize=(12,8))
ax.hist(DMC_err, facecolor='b', edgecolor='black', alpha=0.7, density=True, label=f'Direct Sampling: $\sigma$ = {np.std(DMC_err):.3e}')
ax.hist(IS_err, facecolor='r', edgecolor='black', alpha=0.5, density=True, label=f'Importance Sampling: $\sigma$ = {np.std(IS_err):.3e}')
ax.set_xlabel('Estimation error', fontsize=_font)
ax.set_ylabel('Probability', fontsize=_font)
ax.set_title('Importance Sampling versus Direct Sampling', fontsize=_font + 6)
ax.tick_params('both', labelsize=_font-4)
ax.legend(fontsize=_font)


plt.show()