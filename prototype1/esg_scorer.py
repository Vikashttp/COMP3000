# esg_scorer.py
# Handles ESG scoring, benchmarking, and recommendations for Veridex platform

import pandas as pd
import numpy as np


def score_label(score):
    """Convert a numeric ESG score into a readable rating."""
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Fair"
    else:
        return "Poor"


def compute_custom_esg(row, wE, wS, wG):
    """
    Calculate a weighted ESG total from Environmental, Social, Governance scores.
    Weights are normalised so they always sum to 1.
    This makes the scoring fully transparent and explainable.
    """
    total_w = wE + wS + wG
    if total_w == 0:
        wE, wS, wG = 0.4, 0.3, 0.3
    else:
        wE = wE / total_w
        wS = wS / total_w
        wG = wG / total_w

    e = pd.to_numeric(row.get("Environmental_Score", np.nan), errors="coerce")
    s = pd.to_numeric(row.get("Social_Score", np.nan), errors="coerce")
    g = pd.to_numeric(row.get("Governance_Score", np.nan), errors="coerce")

    if pd.isna(e) or pd.isna(s) or pd.isna(g):
        return np.nan

    return round(e * wE + s * wS + g * wG, 2)


def greenwashing_flag(row):
    """
    Basic greenwashing detection.
    If a company has a high ESG score but also high emissions intensity,
    that is a potential inconsistency worth flagging.
    """
    esg = pd.to_numeric(row.get("Overall_ESG_Score", np.nan), errors="coerce")
    emissions_intensity = pd.to_numeric(row.get("Emissions_Intensity", np.nan), errors="coerce")

    if pd.isna(esg) or pd.isna(emissions_intensity):
        return "No Data"
    if esg >= 75 and emissions_intensity >= 60:
        return "Potential Greenwashing"
    if esg >= 70 and emissions_intensity >= 40:
        return "Watch"
    return "Consistent"


def sector_benchmark(df):
    """
    Calculate average ESG scores per sector.
    Used to compare a company against its industry peers.
    """
    cols = ["Sector", "Overall_ESG_Score", "Environmental_Score",
            "Social_Score", "Governance_Score"]
    existing = [c for c in cols if c in df.columns]
    result = (
        df[existing]
        .groupby("Sector")
        .mean(numeric_only=True)
        .round(2)
        .reset_index()
    )
    return result


def financial_health_score(row):
    """
    Simple financial health indicator based on profit margin.
    """
    net = pd.to_numeric(row.get("Net_Income", np.nan), errors="coerce")
    rev = pd.to_numeric(row.get("Revenue", np.nan), errors="coerce")

    if pd.isna(net) or pd.isna(rev) or rev == 0:
        return "Unknown"

    margin = (net / rev) * 100

    if margin >= 20:
        return "Strong"
    elif margin >= 8:
        return "Moderate"
    elif margin >= 0:
        return "Weak"
    else:
        return "Loss-Making"


def get_recommendations(row):
    """
    Generate improvement suggestions based on weak ESG metrics.
    Each check is based on documented thresholds from ESG literature.
    """
    tips = []

    renewable = pd.to_numeric(row.get("Renewable_Energy_Percentage", np.nan), errors="coerce")
    gender = pd.to_numeric(row.get("Gender_Diversity_Percentage", np.nan), errors="coerce")
    board_ind = pd.to_numeric(row.get("Independent_Board_Percentage", np.nan), errors="coerce")
    injury = pd.to_numeric(row.get("Workplace_Injury_Rate", np.nan), errors="coerce")
    turnover = pd.to_numeric(row.get("Employee_Turnover_Percentage", np.nan), errors="coerce")
    esg_comp = pd.to_numeric(row.get("ESG_Linked_Compensation_Yes_No", np.nan), errors="coerce")
    fines = pd.to_numeric(row.get("Regulatory_Fines", np.nan), errors="coerce")

    if not pd.isna(renewable) and renewable < 50:
        tips.append(f"Increase renewable energy usage (currently {renewable:.0f}%, target 50% or above)")

    if not pd.isna(gender) and gender < 40:
        tips.append(f"Improve gender diversity (currently {gender:.0f}%, target 40% or above)")

    if not pd.isna(board_ind) and board_ind < 70:
        tips.append(f"Increase independent board members (currently {board_ind:.0f}%, target 70% or above)")

    if not pd.isna(injury) and injury > 1.0:
        tips.append(f"Reduce workplace injury rate (currently {injury:.2f}, target below 1.0)")

    if not pd.isna(turnover) and turnover > 20:
        tips.append(f"Address high employee turnover (currently {turnover:.0f}%, target below 20%)")

    if not pd.isna(esg_comp) and esg_comp == 0:
        tips.append("Consider linking executive compensation to ESG performance targets")

    if not pd.isna(fines) and fines > 0:
        tips.append(f"Address regulatory compliance issues (fines recorded: ${fines:,.0f})")

    if not tips:
        tips.append("Strong ESG performance across all measured indicators. Maintain current practices.")

    return tips