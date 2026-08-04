# =============================================================================
# ECON-5371 -- Problem Set 1: Solutions
# Review: Stationarity, ARMA, and ARIMA (Chapters 1-4)
#
# Data: ps1_data.csv
#   confidence_index     -- simulated stationary AR(2) series
#   housing_price_index  -- simulated I(1) series, random walk with drift
#   140 quarterly observations, 1990Q1-2024Q4
#
# True DGP):
#   confidence_index:    y_t - 50 = 0.60*(y_{t-1}-50) - 0.25*(y_{t-2}-50) + e_t,  e_t ~ N(0,1)
#   housing_price_index: y_t = y_{t-1} + 0.15 + e_t,                              e_t ~ N(0,0.8^2)
#
# Figures are written to ./figures/. Run with `python PS1_solution.py`.
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
import warnings

warnings.filterwarnings("ignore")

PATH = "problem_sets/1/"
os.makedirs(PATH, exist_ok=True)

H = 8  # holdout horizon, in quarters. Same holdout is reused in PS2 (Ch5, Diebold-Mariano).

df = pd.read_csv(PATH + "/ps1_data.csv", parse_dates=["date"]).set_index("date")
a = df["confidence_index"]
b = df["housing_price_index"]


# =============================================================================
# A.1 -- Series A: confidence_index (stationary branch)
# =============================================================================

# --- Step 1: plot -----------------------------------------------------------
# Fluctuates around a stable level near 50, no trend, no visible change in
# variance -- consistent with a stationary, mean-reverting process.
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(a.index, a.values)
ax.set_title("Series A: Confidence Index")
ax.set_xlabel("Date")
ax.set_ylabel("Index")
fig.tight_layout()
fig.savefig(PATH + "/A1_series_plot.png", dpi=150)
plt.close(fig)

# --- Step 2: stationarity tests, levels -------------------------------------
adf_a = adfuller(a, autolag="AIC")
kpss_a = kpss(a, regression="c", nlags="auto")
print("Series A -- ADF test   (H0: series has a unit root / is nonstationary)")
print(f"  stat={adf_a[0]:.4f}  p={adf_a[1]:.4f}  lags={adf_a[2]}")
print("Series A -- KPSS test  (H0: series is stationary)")
print(f"  stat={kpss_a[0]:.4f}  p={kpss_a[1]:.4f}")
# ADF rejects the unit-root null decisively; KPSS fails to reject stationarity.
# Both agree: Series A is stationary in levels. No differencing needed.

# --- Step 3: identify model order via ACF/PACF ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_acf(a, lags=12, ax=axes[0])
plot_pacf(a, lags=12, ax=axes[1], method="ywm")
axes[0].set_title("ACF, Series A (levels)")
axes[1].set_title("PACF, Series A (levels)")
fig.tight_layout()
fig.savefig(PATH + "/A1_acf_pacf.png", dpi=150)
plt.close(fig)
# ACF decays gradually (AR signature). PACF has significant spikes at lags
# 1-2 and is negligible after -- textbook AR(2) cutoff. Identification: AR(2).

# --- Step 4: estimate on training data (holdout last H quarters) -----------
train_a, test_a = a.iloc[:-H], a.iloc[-H:]

fit_a = ARIMA(train_a, order=(2, 0, 0), trend="c").fit()
print(fit_a.summary())

# Recovered coefficients vs. true DGP (const=50.00, ar.L1=0.60, ar.L2=-0.25):
# ar.L1 is close to truth. ar.L2 undershoots and is only marginally
# significant (p ~ 0.09) -- expected sampling noise at n=132, not an error.
# Worth flagging for students: correct model order doesn't guarantee precise
# coefficient estimates.
p = fit_a.params
print(f"Estimated: const={p['const']:.2f}  ar.L1={p['ar.L1']:.2f}  ar.L2={p['ar.L2']:.2f}")

# --- Step 5: diagnostics -----------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
plot_acf(fit_a.resid, lags=12, ax=ax)
ax.set_title("Residual ACF, Series A AR(2)")
fig.tight_layout()
fig.savefig(PATH + "/A1_resid_acf.png", dpi=150)
plt.close(fig)

lb_a = acorr_ljungbox(fit_a.resid, lags=[8], return_df=True)
print("Series A -- Ljung-Box test on residuals  (H0: residuals are white noise / no autocorrelation)")
print(lb_a)
# Fails to reject white noise (p ~ 0.83). AR(2) is adequate; no revision needed.

# --- Step 6: forecast --------------------------------------------------------
fc_a = fit_a.get_forecast(steps=H)
fc_mean_a = fc_a.predicted_mean
fc_ci_a = fc_a.conf_int(alpha=0.10)

comp_a = pd.DataFrame({
    "actual": test_a.values,
    "forecast": fc_mean_a.values,
    "lo90": fc_ci_a.iloc[:, 0].values,
    "hi90": fc_ci_a.iloc[:, 1].values,
})
print(comp_a)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_a.index[-20:], train_a.values[-20:], label="Train")
ax.plot(test_a.index, test_a.values, marker="o", label="Actual (holdout)")
ax.plot(test_a.index, fc_mean_a.values, linestyle="--", label="Forecast")
ax.fill_between(test_a.index, fc_ci_a.iloc[:, 0], fc_ci_a.iloc[:, 1],
                 alpha=0.2, label="90% interval")
ax.legend()
ax.set_title("Series A: AR(2) Forecast vs Holdout")
fig.tight_layout()
fig.savefig(PATH + "/A1_forecast.png", dpi=150)
plt.close(fig)
# Forecast reverts quickly to the estimated mean (~49.9) rather than trending
# -- expected for a stationary process. All 8 holdout points fall inside the
# 90% interval.
#
# KEEP fit_a, fc_a, test_a -- PS2 (Ch5) reuses this exact model/forecast for
# the Diebold-Mariano test.


# =============================================================================
# A.2 -- Series B: housing_price_index (nonstationary branch)
# =============================================================================

# --- Step 1: plot -------------------------------------------------------------
# Persistent upward trend, no tendency to revert to a fixed level -- visual
# hallmark of nonstationarity, unlike Series A.
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(b.index, b.values)
ax.set_title("Series B: Housing Price Index")
ax.set_xlabel("Date")
ax.set_ylabel("Index")
fig.tight_layout()
fig.savefig(PATH + "/A2_series_plot.png", dpi=150)
plt.close(fig)

# --- Step 2a: stationarity tests, levels --------------------------------------
adf_b_lvl = adfuller(b, autolag="AIC")
kpss_b_lvl = kpss(b, regression="ct", nlags="auto")
print("Series B (levels) -- ADF test   (H0: series has a unit root / is nonstationary)")
print(f"  stat={adf_b_lvl[0]:.4f}  p={adf_b_lvl[1]:.4f}")
print("Series B (levels) -- KPSS test  (H0: series is stationary)")
print(f"  stat={kpss_b_lvl[0]:.4f}  p={kpss_b_lvl[1]:.4f}")
# ADF fails to reject the unit-root null (p ~ 0.98). KPSS rejects stationarity
# (p ~ 0.01). Both point the same direction: nonstationary in levels. Difference
# and re-test.

# --- Step 2b: stationarity tests, first difference ----------------------------
db = b.diff().dropna()
adf_b_diff = adfuller(db, autolag="AIC")
kpss_b_diff = kpss(db, regression="c", nlags="auto")
print("Series B (first diff) -- ADF test   (H0: series has a unit root / is nonstationary)")
print(f"  stat={adf_b_diff[0]:.4f}  p={adf_b_diff[1]:.4f}")
print("Series B (first diff) -- KPSS test  (H0: series is stationary)")
print(f"  stat={kpss_b_diff[0]:.4f}  p={kpss_b_diff[1]:.4f}")
print(f"Mean of diff(b), i.e. estimated drift: {db.mean():.4f}  (true drift = 0.15)")
# Both tests flip relative to levels -- ADF now rejects, KPSS now fails to
# reject. Confirms Series B is I(1).

# --- Step 3: identify model order on the differenced series -------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_acf(db, lags=12, ax=axes[0])
plot_pacf(db, lags=12, ax=axes[1], method="ywm")
axes[0].set_title("ACF, diff(Series B)")
axes[1].set_title("PACF, diff(Series B)")
fig.tight_layout()
fig.savefig(PATH + "/A2_acf_pacf_diff.png", dpi=150)
plt.close(fig)
# No significant spikes at any lag -- the differenced series is white noise.
# Identification: levels series is ARIMA(0,1,0), a random walk with drift.

# --- Step 4: estimate -----------------------------------------------------
train_b, test_b = b.iloc[:-H], b.iloc[-H:]

fit_b = ARIMA(train_b, order=(0, 1, 0), trend="t").fit()
print(fit_b.summary())
print(f"Estimated drift: {fit_b.params['x1']:.4f}  (true drift = 0.15)")

# Cross-check against auto_arima. Use it to confirm the manual Box-Jenkins
# identification, not to replace it (Ch4's stated limitation).
auto_b = pm.auto_arima(train_b, seasonal=False, trace=False, suppress_warnings=True)
print(auto_b.summary())
# auto_arima independently selects ARIMA(0,1,0) -- agrees with manual ID.

# --- Step 5: diagnostics -----------------------------------------------------
lb_b = acorr_ljungbox(fit_b.resid[1:], lags=[8], return_df=True)
print("Series B -- Ljung-Box test on residuals  (H0: residuals are white noise / no autocorrelation)")
print(lb_b)
# Fails to reject white noise (p ~ 0.56). Residuals OK.

# --- Step 6: forecast --------------------------------------------------------
fc_b = fit_b.get_forecast(steps=H)
fc_mean_b = fc_b.predicted_mean
fc_ci_b = fc_b.conf_int(alpha=0.10)

comp_b = pd.DataFrame({
    "actual": test_b.values,
    "forecast": fc_mean_b.values,
    "lo90": fc_ci_b.iloc[:, 0].values,
    "hi90": fc_ci_b.iloc[:, 1].values,
})
print(comp_b)

w1 = fc_ci_b.iloc[0, 1] - fc_ci_b.iloc[0, 0]
w8 = fc_ci_b.iloc[-1, 1] - fc_ci_b.iloc[-1, 0]
print(f"90% interval width -- h=1: {w1:.3f}   h=8: {w8:.3f}   ratio: {w8 / w1:.2f}x")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_b.index[-20:], train_b.values[-20:], label="Train")
ax.plot(test_b.index, test_b.values, marker="o", label="Actual (holdout)")
ax.plot(test_b.index, fc_mean_b.values, linestyle="--", label="Forecast")
ax.fill_between(test_b.index, fc_ci_b.iloc[:, 0], fc_ci_b.iloc[:, 1],
                 alpha=0.2, label="90% interval")
ax.legend()
ax.set_title("Series B: ARIMA(0,1,0) Forecast vs Holdout")
fig.tight_layout()
fig.savefig(PATH + "/A2_forecast.png", dpi=150)
plt.close(fig)
# Point forecast is a straight line (slope = drift) rather than reverting --
# an I(1) process has no mean to revert to. Interval nearly triples from h=1
# to h=8, illustrating Ch4's sqrt(h) interval-growth result for integrated
# processes, vs. the ceiling Series A converges to.