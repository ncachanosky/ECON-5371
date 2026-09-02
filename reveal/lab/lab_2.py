"""
ECON-5371 — Lab 1: Non-Stationary Models, Seasonality
Week 5

Applies unit root testing (ADF, KPSS), ARIMA identification, SARIMA
estimation, and residual diagnostics to a synthetic quarterly series
with a stochastic trend and seasonal pattern. Forecasting is covered
in Lab 2.

Before running: make sure gdp_synthetic.csv has been downloaded from
Blackboard, placed in your lab_1/ folder, and pushed to your own
GitHub repository. Replace the URL below with YOUR OWN raw GitHub
URL — this one points to the instructor's copy of the data, not
yours.
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# tab10 is matplotlib's built-in qualitative palette -- the same one it uses
# automatically when you plot multiple series without specifying colors.
# Referencing it explicitly here just makes that choice visible and
# consistent across every plot in this script.
tab10 = plt.get_cmap("tab10")
COLOR_OBSERVED = tab10(0)  # blue

# =============================================================================
# 1. Load the data from GitHub
# =============================================================================
# Reading directly from a raw GitHub URL — rather than a local file path —
# means this script reproduces the exact same result for anyone who runs it,
# without needing a copy of the CSV sitting on their own machine.
#
# REPLACE THIS with your own repo's raw URL (github.com file page -> "Raw"
# button). Using the instructor's URL here will read the instructor's data,
# not the copy you committed yourself.
url = "https://raw.githubusercontent.com/ncachanosky/ECON-5371-lab/main/lab_1/gdp_synthetic.csv"
df = pd.read_csv(url, parse_dates=["date"])
df = df.set_index("date")
df.index.freq = "QS"   # explicitly quarterly-start; avoids statsmodels having
                       # to guess the frequency (and warn about it) every time

print(df.head())


# =============================================================================
# 2. Plot the raw series
# =============================================================================
# Always look before testing. A visible trend and a repeating pattern here
# are exactly what we'd expect to confirm formally in the next section.
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(df.index, df["business_activity_index"], color=COLOR_OBSERVED, linewidth=1.6)
ax.set_title("Business Activity Index — Synthetic Quarterly Series")
ax.set_xlim(df.index.min(), df.index.max())
fig.tight_layout()
plt.show()


# =============================================================================
# 3. Unit root testing on the level series
# =============================================================================
# Running both ADF and KPSS gives a joint check rather than relying on a
# single test -- their null hypotheses point in opposite directions, so
# agreement between them is a stronger signal than either test alone.

print("ADF test")
print("H0: the series has a unit root (non-stationary)")
adf_level = adfuller(df["business_activity_index"], autolag="AIC")
print(f"  statistic = {adf_level[0]:.4f}")
print(f"  p-value   = {adf_level[1]:.4f}")
print(f"  lags used = {adf_level[2]}")
# Fails to reject H0 (p = 0.9480) -> consistent with a unit root.

print()
print("KPSS test")
print("H0: the series is stationary")
kpss_level = kpss(df["business_activity_index"], regression="c", nlags="auto")
print(f"  statistic = {kpss_level[0]:.4f}")
print(f"  p-value   = {kpss_level[1]:.4f}")
# Rejects H0 (p = 0.0100) -> the series is NOT stationary.
#
# Both tests agree: this is a genuine unit-root series, not a borderline case.


# =============================================================================
# 4. Difference the series and re-test
# =============================================================================
# .diff() computes y_t - y_(t-1). The first observation becomes missing by
# construction, so we drop it before re-testing.
df["diff"] = df["business_activity_index"].diff()
df_diff = df["diff"].dropna()

print(df_diff.head())

print()
print("Re-testing after one difference")
print("ADF H0: unit root  |  KPSS H0: stationary")
adf_diff = adfuller(df_diff, autolag="AIC")
kpss_diff = kpss(df_diff, regression="c", nlags="auto")

print(f"  ADF:  statistic = {adf_diff[0]:.4f}, p-value = {adf_diff[1]:.4f}")
print(f"  KPSS: statistic = {kpss_diff[0]:.4f}, p-value = {kpss_diff[1]:.4f}")
# After one difference: ADF rejects H0 (p = 0.0344), KPSS fails to reject
# (p >= 0.10). Both tests flip and agree -> the level series is I(1).
#
# Note on the KPSS warning: statsmodels may print an InterpolationWarning
# here. That's not an error -- the test statistic falls outside the range
# covered by KPSS's lookup table, so it can only report a bound on the
# p-value (e.g. "p >= 0.10") rather than an exact figure. Given how clearly
# stationary this differenced series is, that's expected, not a problem.


# =============================================================================
# 5. Plot the differenced series
# =============================================================================
# Confirm visually what the tests just told us: the trend should be gone,
# but if seasonality is still present, we'd expect to still see a repeating
# wiggle in this plot -- differencing once removes the trend, not the
# seasonal pattern.
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(df_diff.index, df_diff, color=COLOR_OBSERVED, linewidth=1.4)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_title("First-Differenced Business Activity Index")
ax.set_xlim(df_diff.index.min(), df_diff.index.max())
fig.tight_layout()
plt.show()
# No more trend -- the series now fluctuates around zero. The repeating
# up-down pattern that remains is the seasonal signal, which the ACF/PACF
# below will help us characterize precisely.


# =============================================================================
# 6. ACF / PACF on the differenced series
# =============================================================================
# Dotted vertical lines mark the seasonal lags (4, 8, 12) where we'd expect
# a quarterly pattern to show up, if one is present.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
plot_acf(df_diff, lags=12, ax=axes[0])
plot_pacf(df_diff, lags=12, ax=axes[1])
for ax in axes:
    for seasonal_lag in [4, 8, 12]:
        ax.axvline(seasonal_lag, color="gray", linestyle=":", linewidth=1)
fig.tight_layout()
plt.show()
# Sharp ACF spikes recur every 4 lags (~0.87, 0.84, 0.77 at lags 4, 8, 12).
# A short-run ARMA pattern would fade out gradually; this isolated, repeating
# spike is the fingerprint of quarterly seasonality -- this series needs a
# seasonal term, not just a plain ARMA.


# =============================================================================
# 7. Fit a first candidate ARIMA (no seasonal term, for comparison)
# =============================================================================
candidate = ARIMA(df["business_activity_index"], order=(1, 1, 1), freq="QS")
candidate_fit = candidate.fit()

print(f"Candidate ARIMA(1,1,1) AIC: {candidate_fit.aic:.3f}")
print(f"Candidate ARIMA(1,1,1) BIC: {candidate_fit.bic:.3f}")
# Keep these numbers to compare against the seasonal specification below.


# =============================================================================
# 8. Let auto_arima search over seasonal specifications
# =============================================================================
# auto_arima searches efficiently over a grid of specifications -- it does
# not know economics. Always sanity-check its result against what the
# ACF/PACF showed, the way we just did above.
auto_model = pm.auto_arima(
    df["business_activity_index"],
    seasonal=True,
    m=4,  # quarterly seasonal period
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
)

print(auto_model.summary())
# auto_arima selects SARIMA(1,0,1)(0,1,1)[4], AIC = 193.67 -- clearly better
# than the non-seasonal ARIMA(1,1,1) candidate above.


# =============================================================================
# 9. Fit the selected SARIMA model directly
# =============================================================================
# (1,0,1) non-seasonal: short-run AR(1)/MA(1) dynamics quarter to quarter.
# (0,1,1) seasonal: one seasonal difference removes the repeating pattern;
#   one seasonal MA term captures leftover seasonal correlation.
# [4]: the seasonal period is 4 quarters.
final_model = SARIMAX(
    df["business_activity_index"],
    order=(1, 0, 1),
    seasonal_order=(0, 1, 1, 4),
    freq="QS",
)
final_fit = final_model.fit(disp=False)

print(final_fit.summary())
# AR(1) coefficient ~0.620 (significant), seasonal MA coefficient ~-0.803
# (significant) -- both terms are doing real work, not just padding the
# specification.
#
# Note: the summary table above already reports Ljung-Box(L1) and
# Jarque-Bera as quick built-in checks. Section 10 below runs a fuller
# Ljung-Box test over several lags and looks at the residuals directly.


# =============================================================================
# 10. Residual diagnostics
# =============================================================================
# A well-specified model should leave behind residuals that look like white
# noise -- no leftover pattern, no leftover autocorrelation. If diagnostics
# below fail, that's a sign the model missed something worth revisiting
# before trusting its forecast.
#
# We burn the first 8 residuals (two full seasonal cycles) before testing.
# The state-space filter needs a few observations to initialize; residuals
# from that startup window are not genuine one-step-ahead forecast errors
# and will swamp any diagnostic test if left in. This is the same
# burn-in principle used elsewhere in this course for presample outliers.
residuals = final_fit.resid.iloc[8:]

print("Ljung-Box test on residuals")
print("H0: residuals are not autocorrelated (i.e., they look like white noise)")
ljung_box = acorr_ljungbox(residuals, lags=[4, 8, 12], return_df=True)
print(ljung_box)
# All three p-values are well above 0.05 (0.59, 0.59, 0.56) -> fail to
# reject H0. The residuals show no significant leftover autocorrelation at
# the seasonal lags -- a good sign the seasonal structure was fully
# captured by the model.

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(residuals.index, residuals, color=COLOR_OBSERVED, linewidth=1.2)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_title("SARIMA Residuals (post burn-in)")
ax.set_xlim(residuals.index.min(), residuals.index.max())
fig.tight_layout()
plt.show()
# Residuals scatter randomly around zero with no visible trend or repeating
# pattern -- consistent with the Ljung-Box result above.

fig, ax = plt.subplots(figsize=(6, 3.5))
plot_acf(residuals, lags=12, ax=ax)
for seasonal_lag in [4, 8, 12]:
    ax.axvline(seasonal_lag, color="gray", linestyle=":", linewidth=1)
ax.set_title("ACF of Residuals")
fig.tight_layout()
plt.show()
# No spikes extend beyond the shaded confidence band, including at the
# seasonal lags marked above. This confirms the same conclusion as the
# Ljung-Box test: the residuals are consistent with white noise.
#
# The model has been identified, estimated, and diagnosed -- a complete
# Box-Jenkins pass. Forecasting with this model is the starting point for
# Lab 2.