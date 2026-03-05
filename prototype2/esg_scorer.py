# esg_scorer.py
# Handles ESG scoring, greenwashing detection, sector benchmarking
# and company recommendations for Veridex

import pandas as pd


def calculate_weighted_esg(env, social, gov, env_w, soc_w, gov_w):
    # calculates a custom ESG score based on how much each pillar matters
    # this is the transparency feature - unlike MSCI who hide their weights
    total = env_w + soc_w + gov_w
    if total == 0:
        return 0.0
    return round(((env * env_w) + (social * soc_w) + (gov * gov_w)) / total, 1)


def get_esg_rating(score):
    # simple rating system - like a school grade
    # makes it easy for anyone to understand the score at a glance
    if score >= 80:
        return "Excellent", "#00d4aa"
    elif score >= 65:
        return "Good", "#0ea5e9"
    elif score >= 50:
        return "Fair", "#f59e0b"
    else:
        return "Poor", "#ef4444"


def get_sector_average(df, sector, metric="Overall_ESG_Score"):
    # compares one company against others in the same industry
    sector_df = df[df["Sector"] == sector]
    if sector_df.empty:
        return None
    return round(sector_df[metric].mean(), 1)
import os
def check_greenwashing(company_data):
    # compares ESG score against actual emissions
    # a company with high ESG but very high emissions might be exaggerating
    esg_score = company_data.get("Overall_ESG_Score", 50)
    emissions  = company_data.get("Total_Emissions", 0)
    revenue    = company_data.get("Revenue", 1)

    if revenue <= 0:
        revenue = 1

    intensity = emissions / revenue

    if esg_score >= 70 and intensity > 50:
        return (
            "ALERT: High ESG score but emissions are very high relative to revenue. Independent verification is recommended.",
            "#ef4444"
        )
    elif esg_score >= 70 and intensity > 20:
        return (
            "WARNING: ESG score looks good but emissions are above sector norms. Worth monitoring closely.",
            "#f59e0b"
        )
    else:
        return (
            "CONSISTENT: ESG score aligns with actual emissions data. No greenwashing indicators found.",
            "#00d4aa"
        )
import google.generativeai as genai

# load the Gemini API key from .env file
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_recommendations(company_name, company_data, sector, sector_avg):
    # sends real company data to Gemini and gets unique AI written tips back
    # every company gets different recommendations based on their actual numbers
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""You are an ESG analyst. Give exactly 4 short, specific improvement recommendations for {company_name}.

Company data:
- Sector: {sector}
- Overall ESG Score: {company_data.get('Overall_ESG_Score', 'N/A')}
- Renewable Energy: {company_data.get('Renewable_Energy_Percentage', 'N/A')}%
- Gender Diversity: {company_data.get('Gender_Diversity_Percentage', 'N/A')}%
- Employee Turnover: {company_data.get('Employee_Turnover_Percentage', 'N/A')}%
- Training Hours: {company_data.get('Training_Hours_Per_Employee', 'N/A')} hrs/year
- Independent Board: {company_data.get('Independent_Board_Percentage', 'N/A')}%
- Workplace Injury Rate: {company_data.get('Workplace_Injury_Rate', 'N/A')}
- Carbon Reduction Target: {company_data.get('Carbon_Reduction_Target_Percentage', 'N/A')}%
- Sector Average ESG: {sector_avg}

Rules:
- Reference the actual numbers above
- Be specific not generic
- No bullet points or numbering, just plain sentences
- One sentence per recommendation
- Return exactly 4 recommendations separated by newlines"""

        response = model.generate_content(prompt)
        lines = [l.strip() for l in response.text.strip().split('\n') if l.strip()]
        return lines[:4]

    except Exception:
        # if API fails for any reason fall back to data driven tips
        return get_fallback_recommendations(company_name, company_data, sector, sector_avg)


def get_fallback_recommendations(company_name, company_data, sector, sector_avg):
    # backup tips using real numbers if Gemini is unavailable
    recs = []
    renewable = company_data.get("Renewable_Energy_Percentage", 50)
    diversity = company_data.get("Gender_Diversity_Percentage", 40)
    board     = company_data.get("Independent_Board_Percentage", 70)
    training  = company_data.get("Training_Hours_Per_Employee", 30)
    esg_comp  = company_data.get("ESG_Linked_Compensation_Yes_No", 0)
    injury    = company_data.get("Workplace_Injury_Rate", 1)
    esg_score = company_data.get("Overall_ESG_Score", 50)
    carbon    = company_data.get("Carbon_Reduction_Target_Percentage", 20)

    if renewable < 50:
        recs.append(f"Increasing renewable energy from {renewable:.0f}% toward 50% would have the biggest impact on the Environmental score.")
    if diversity < 40:
        recs.append(f"Gender diversity at {diversity:.0f}% is below the 40% benchmark -- structured hiring targets would improve the Social score.")
    if board < 70:
        recs.append(f"Board independence at {board:.0f}% is below best practice -- appointing more independent directors strengthens Governance.")
    if training < 30:
        recs.append(f"Training at {training:.0f} hours per employee is low -- companies investing 40+ hours see measurably lower staff turnover.")
    if esg_comp == 0 or esg_comp == "No":
        recs.append("Linking executive pay to ESG targets is standard among top performers and signals genuine sustainability commitment.")
    if injury > 2:
        recs.append(f"A workplace injury rate of {injury:.1f} is above acceptable thresholds -- safety programme investment would directly lift the Social score.")
    if sector_avg and esg_score < sector_avg:
        recs.append(f"{company_name} scores {round(sector_avg - esg_score, 1)} points below the {sector} sector average -- closing this gap should be the near-term priority.")
    if carbon < 30:
        recs.append(f"A carbon reduction target of only {carbon:.0f}% is unambitious by current standards -- raising this signals long-term climate commitment.")
    if not recs:
        recs.append(f"{company_name} is performing well. Maintaining third-party ESG audits will preserve investor confidence.")

    return recs[:4]