import math

def calculate_framingham_10yr_risk(age, sex, total_cholesterol, hdl_cholesterol, systolic_bp, smoker, diabetes, bp_meds):
    """
    A simplified synthetic Framingham 10-year cardiovascular risk calculator.
    This is a mocked heuristic approximation for demonstration purposes.
    Returns the risk percentage and a risk tier.
    """
    risk_score = 0
    
    # 1. Age factor
    if age > 70:
        risk_score += 10
    elif age > 60:
        risk_score += 8
    elif age > 50:
        risk_score += 5
    elif age > 40:
        risk_score += 2
        
    # 2. Cholesterol factor
    if total_cholesterol > 240:
        risk_score += 5
    elif total_cholesterol > 200:
        risk_score += 3
        
    if hdl_cholesterol < 40:
        risk_score += 2
    elif hdl_cholesterol >= 60:
        risk_score -= 1
        
    # 3. Blood Pressure
    if systolic_bp > 160:
        risk_score += 5 if not bp_meds else 4
    elif systolic_bp > 140:
        risk_score += 3 if not bp_meds else 2
    elif systolic_bp > 130:
        risk_score += 1 if not bp_meds else 0
        
    # 4. Lifestyle and Comorbidities
    if smoker:
        risk_score += 6
        
    if diabetes:
        risk_score += 5
        
    # Base risk math mapping (synthetic)
    base_calc = max(1.0, math.exp(risk_score * 0.15))
    if sex.lower() == 'male':
        risk_percent = min(100.0, base_calc * 1.5)
    else:
        risk_percent = min(100.0, base_calc * 1.1)
        
    risk_percent = round(risk_percent, 1)
    
    # Risk Tier
    if risk_percent < 5.0:
        tier = "Low"
    elif risk_percent < 7.5:
        tier = "Borderline"
    elif risk_percent < 20.0:
        tier = "Intermediate"
    else:
        tier = "High"
        
    return {
        "risk_percentage": risk_percent,
        "risk_tier": tier
    }
