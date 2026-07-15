import math

# 2013 ACC/AHA Pooled Cohort Equations (Goff DC et al., Circulation 2014;129(25 Suppl 2):S49-73).
# Coefficients are the published race/sex-specific values validated against the
# worked example in the guideline's supplement (55yo, TC 213, HDL 50, SBP 120
# untreated, nonsmoker, no diabetes -> White woman 2.1%, Black woman 3.0%,
# White man 5.4%, Black man 6.1%).
_COEFFICIENTS = {
    ("female", "black"): dict(
        base_survival=0.95334, mean=86.6081,
        ln_age=17.1141, ln_age_sq=0.0,
        ln_tc=0.9396, ln_age_x_ln_tc=0.0,
        ln_hdl=-18.9196, ln_age_x_ln_hdl=4.4748,
        ln_sbp_treated=29.2907, ln_age_x_ln_sbp_treated=-6.4321,
        ln_sbp_untreated=27.8197, ln_age_x_ln_sbp_untreated=-6.0873,
        smoker=0.6908, ln_age_x_smoker=0.0,
        diabetes=0.8738,
    ),
    ("female", "white"): dict(
        base_survival=0.96652, mean=-29.1817,
        ln_age=-29.799, ln_age_sq=4.884,
        ln_tc=13.54, ln_age_x_ln_tc=-3.114,
        ln_hdl=-13.578, ln_age_x_ln_hdl=3.149,
        ln_sbp_treated=2.019, ln_age_x_ln_sbp_treated=0.0,
        ln_sbp_untreated=1.957, ln_age_x_ln_sbp_untreated=0.0,
        smoker=7.574, ln_age_x_smoker=-1.665,
        diabetes=0.661,
    ),
    ("male", "black"): dict(
        base_survival=0.89536, mean=19.5425,
        ln_age=2.469, ln_age_sq=0.0,
        ln_tc=0.302, ln_age_x_ln_tc=0.0,
        ln_hdl=-0.307, ln_age_x_ln_hdl=0.0,
        ln_sbp_treated=1.916, ln_age_x_ln_sbp_treated=0.0,
        ln_sbp_untreated=1.809, ln_age_x_ln_sbp_untreated=0.0,
        smoker=0.549, ln_age_x_smoker=0.0,
        diabetes=0.645,
    ),
    ("male", "white"): dict(
        base_survival=0.91436, mean=61.1816,
        ln_age=12.344, ln_age_sq=0.0,
        ln_tc=11.853, ln_age_x_ln_tc=-2.664,
        ln_hdl=-7.99, ln_age_x_ln_hdl=1.769,
        ln_sbp_treated=1.797, ln_age_x_ln_sbp_treated=0.0,
        ln_sbp_untreated=1.764, ln_age_x_ln_sbp_untreated=0.0,
        smoker=7.837, ln_age_x_smoker=-1.795,
        diabetes=0.658,
    ),
}


def calculate_ascvd_10yr_risk(age, sex, total_cholesterol, hdl_cholesterol, systolic_bp,
                               smoker, diabetes, bp_meds, race="white"):
    """
    10-year atherosclerotic cardiovascular disease (ASCVD) risk via the
    2013 ACC/AHA Pooled Cohort Equations. Validated for non-Hispanic White
    and African American adults aged 40-79 without prior ASCVD; other
    inputs fall outside the derivation cohort and the result is flagged.
    """
    sex_key = "male" if str(sex).strip().lower().startswith("m") else "female"
    race_normalized = str(race).strip().lower()
    race_key = "black" if race_normalized in ("black", "african_american", "african american") else "white"
    race_is_approximated = race_normalized not in ("white", "black", "african_american", "african american")

    c = _COEFFICIENTS[(sex_key, race_key)]

    ln_age = math.log(age)
    ln_tc = math.log(total_cholesterol)
    ln_hdl = math.log(hdl_cholesterol)
    ln_sbp = math.log(systolic_bp)
    smoker_flag = 1.0 if smoker else 0.0
    diabetes_flag = 1.0 if diabetes else 0.0
    ln_sbp_treated = ln_sbp if bp_meds else 0.0
    ln_sbp_untreated = 0.0 if bp_meds else ln_sbp

    individual_sum = (
        c["ln_age"] * ln_age
        + c["ln_age_sq"] * ln_age ** 2
        + c["ln_tc"] * ln_tc
        + c["ln_age_x_ln_tc"] * ln_age * ln_tc
        + c["ln_hdl"] * ln_hdl
        + c["ln_age_x_ln_hdl"] * ln_age * ln_hdl
        + c["ln_sbp_treated"] * ln_sbp_treated
        + c["ln_age_x_ln_sbp_treated"] * ln_age * ln_sbp_treated
        + c["ln_sbp_untreated"] * ln_sbp_untreated
        + c["ln_age_x_ln_sbp_untreated"] * ln_age * ln_sbp_untreated
        + c["smoker"] * smoker_flag
        + c["ln_age_x_smoker"] * ln_age * smoker_flag
        + c["diabetes"] * diabetes_flag
    )

    risk_fraction = 1 - c["base_survival"] ** math.exp(individual_sum - c["mean"])
    risk_percent = round(max(0.0, min(100.0, risk_fraction * 100)), 1)

    if risk_percent < 5.0:
        tier = "Low"
    elif risk_percent < 7.5:
        tier = "Borderline"
    elif risk_percent < 20.0:
        tier = "Intermediate"
    else:
        tier = "High"

    out_of_range = age < 40 or age > 79

    return {
        "risk_percentage": risk_percent,
        "risk_tier": tier,
        "method": "2013 ACC/AHA Pooled Cohort Equations",
        "out_of_validated_age_range": out_of_range,
        "race_approximated_as_white": race_is_approximated,
    }
