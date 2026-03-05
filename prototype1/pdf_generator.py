# pdf_generator.py
# Generates a downloadable PDF report for a selected company
# Uses the fpdf2 library which we installed earlier

from fpdf import FPDF
import pandas as pd
import numpy as np
from datetime import date
from esg_scorer import score_label, get_recommendations, financial_health_score


class VeridexReport(FPDF):
    """
    Custom PDF class that extends FPDF with Veridex branding.
    Header and footer are automatically added to every page.
    """

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(0, 150, 136)
        self.cell(0, 10, "Veridex ESG Intelligence Platform", align="L")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Generated: {date.today().strftime('%d %B %Y')}", align="R")
        self.ln(8)
        self.set_draw_color(0, 150, 136)
        self.set_line_width(0.5)
        self.line(10, 22, 200, 22)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Veridex Platform  |  Confidential  |  Page {self.page_no()}", align="C")

    def section_title(self, title):
        """Add a styled section heading."""
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 150, 136)
        self.ln(4)
        self.cell(0, 8, title, ln=True)
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def key_value_row(self, label, value, highlight=False):
        """Add a label/value row, optionally highlighted."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        if highlight:
            self.set_fill_color(240, 248, 246)
            self.cell(90, 7, label, fill=True)
        else:
            self.cell(90, 7, label)

        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        if highlight:
            self.cell(0, 7, str(value), fill=True, ln=True)
        else:
            self.cell(0, 7, str(value), ln=True)


def safe_val(value, fmt=None, suffix=""):
    """
    Safely format a value for display.
    Returns N/A if the value is missing or not a number.
    """
    try:
        v = float(value)
        if np.isnan(v):
            return "N/A"
        if fmt:
            return fmt.format(v) + suffix
        return str(round(v, 2)) + suffix
    except Exception:
        return "N/A"


def fmt_billions(value):
    """Format large financial numbers into readable billions or millions."""
    try:
        v = float(value)
        if np.isnan(v):
            return "N/A"
        if abs(v) >= 1e12:
            return f"${v/1e12:.2f}T"
        elif abs(v) >= 1e9:
            return f"${v/1e9:.2f}B"
        elif abs(v) >= 1e6:
            return f"${v/1e6:.2f}M"
        else:
            return f"${v:,.0f}"
    except Exception:
        return "N/A"


def generate_report(df, company_name):
    """
    Main function that builds the full PDF report for a company.
    
    What it includes:
    - Company overview and ESG score summary
    - Environmental, Social, Governance breakdown
    - Financial health summary
    - Improvement recommendations
    
    Returns the PDF as bytes so Streamlit can offer it as a download.
    """

    # Get the most recent year of data for this company
    company_data = df[df["Company"] == company_name].copy()
    company_data["Year"] = pd.to_numeric(company_data["Year"], errors="coerce")
    company_data = company_data.sort_values("Year")

    if company_data.empty:
        return None

    latest = company_data.iloc[-1]
    sector = str(latest.get("Sector", "N/A"))
    year = str(int(latest.get("Year", 0))) if not pd.isna(latest.get("Year")) else "N/A"

    # ESG scores
    overall_esg = pd.to_numeric(latest.get("Overall_ESG_Score", np.nan), errors="coerce")
    env_score = pd.to_numeric(latest.get("Environmental_Score", np.nan), errors="coerce")
    soc_score = pd.to_numeric(latest.get("Social_Score", np.nan), errors="coerce")
    gov_score = pd.to_numeric(latest.get("Governance_Score", np.nan), errors="coerce")

    rating = score_label(overall_esg) if not pd.isna(overall_esg) else "N/A"
    fin_health = financial_health_score(latest)
    recommendations = get_recommendations(latest)

    # Build the PDF
    pdf = VeridexReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Company title ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 12, company_name, ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Sector: {sector}   |   Report Year: {year}   |   Financial Health: {fin_health}", ln=True)
    pdf.ln(4)

    # ── ESG Score Summary ─────────────────────────────────────────────────────
    pdf.section_title("ESG Score Summary")

    pdf.set_font("Helvetica", "B", 32)
    if not pd.isna(overall_esg):
        if overall_esg >= 85:
            pdf.set_text_color(0, 180, 0)
        elif overall_esg >= 70:
            pdf.set_text_color(200, 160, 0)
        elif overall_esg >= 55:
            pdf.set_text_color(220, 120, 0)
        else:
            pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 14, f"{overall_esg:.1f} / 100  ({rating})", ln=True)
    else:
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 14, "Score not available", ln=True)

    pdf.set_text_color(30, 30, 30)
    pdf.ln(2)

    # ESG component scores
    pdf.set_font("Helvetica", "", 11)
    pdf.key_value_row("Environmental Score", safe_val(env_score, suffix=" / 100"), highlight=True)
    pdf.key_value_row("Social Score", safe_val(soc_score, suffix=" / 100"))
    pdf.key_value_row("Governance Score", safe_val(gov_score, suffix=" / 100"), highlight=True)
    pdf.ln(2)

    # Scoring methodology note
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5,
        "Scoring methodology: Overall ESG Score is a weighted composite of Environmental (40%), "
        "Social (30%), and Governance (30%) scores, calculated using Veridex transparent scoring engine."
    )
    pdf.ln(4)

    # ── Environmental Detail ──────────────────────────────────────────────────
    pdf.section_title("Environmental Performance")
    pdf.set_text_color(30, 30, 30)

    pdf.key_value_row("Total Emissions (tCO2e)",
        safe_val(latest.get("Total_Emissions"), fmt="{:,.0f}"), highlight=True)
    pdf.key_value_row("Scope 1 Emissions",
        safe_val(latest.get("Scope_1_Emissions"), fmt="{:,.0f}"))
    pdf.key_value_row("Scope 2 Emissions",
        safe_val(latest.get("Scope_2_Emissions"), fmt="{:,.0f}"), highlight=True)
    pdf.key_value_row("Scope 3 Emissions",
        safe_val(latest.get("Scope_3_Emissions"), fmt="{:,.0f}"))
    pdf.key_value_row("Renewable Energy Percentage",
        safe_val(latest.get("Renewable_Energy_Percentage"), fmt="{:.1f}", suffix="%"), highlight=True)
    pdf.key_value_row("Emissions Intensity",
        safe_val(latest.get("Emissions_Intensity"), fmt="{:.2f}"))
    pdf.key_value_row("Carbon Reduction Target",
        safe_val(latest.get("Carbon_Reduction_Target_Percentage"), fmt="{:.1f}", suffix="%"), highlight=True)
    pdf.ln(2)

    # ── Social Detail ─────────────────────────────────────────────────────────
    pdf.section_title("Social Performance")

    pdf.key_value_row("Gender Diversity",
        safe_val(latest.get("Gender_Diversity_Percentage"), fmt="{:.1f}", suffix="%"), highlight=True)
    pdf.key_value_row("Board Diversity",
        safe_val(latest.get("Board_Diversity_Percentage"), fmt="{:.1f}", suffix="%"))
    pdf.key_value_row("Employee Count",
        safe_val(latest.get("Employee_Count"), fmt="{:,.0f}"), highlight=True)
    pdf.key_value_row("Employee Turnover",
        safe_val(latest.get("Employee_Turnover_Percentage"), fmt="{:.1f}", suffix="%"))
    pdf.key_value_row("Training Hours Per Employee",
        safe_val(latest.get("Training_Hours_Per_Employee"), fmt="{:.1f}", suffix=" hrs"), highlight=True)
    pdf.key_value_row("Workplace Injury Rate",
        safe_val(latest.get("Workplace_Injury_Rate"), fmt="{:.2f}"))
    pdf.ln(2)

    # ── Governance Detail ─────────────────────────────────────────────────────
    pdf.section_title("Governance Performance")

    pdf.key_value_row("Independent Board Members",
        safe_val(latest.get("Independent_Board_Percentage"), fmt="{:.1f}", suffix="%"), highlight=True)
    pdf.key_value_row("Audit Committee Independence",
        safe_val(latest.get("Audit_Committee_Independence_Percentage"), fmt="{:.1f}", suffix="%"))
    pdf.key_value_row("ESG-Linked Compensation",
        "Yes" if pd.to_numeric(latest.get("ESG_Linked_Compensation_Yes_No", 0), errors="coerce") == 1 else "No",
        highlight=True)
    pdf.key_value_row("Regulatory Fines",
        fmt_billions(latest.get("Regulatory_Fines")))
    pdf.ln(2)

    # ── Financial Summary ─────────────────────────────────────────────────────
    pdf.section_title("Financial Summary")

    pdf.key_value_row("Revenue", fmt_billions(latest.get("Revenue")), highlight=True)
    pdf.key_value_row("Net Income", fmt_billions(latest.get("Net_Income")))
    pdf.key_value_row("Operating Cash Flow", fmt_billions(latest.get("Operating_Cash_Flow")), highlight=True)
    pdf.key_value_row("Market Capitalisation", fmt_billions(latest.get("Market_Cap")))
    pdf.key_value_row("Total Debt", fmt_billions(latest.get("Total_Debt")), highlight=True)
    pdf.key_value_row("Sustainability Investment", fmt_billions(latest.get("Sustainability_Investment")))
    pdf.ln(2)

    # ── Recommendations ───────────────────────────────────────────────────────
    pdf.section_title("Improvement Recommendations")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)

    for i, tip in enumerate(recommendations, 1):
        pdf.cell(8, 7, f"{i}.")
        pdf.multi_cell(0, 7, tip)

    pdf.ln(4)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5,
        "Disclaimer: This report is generated by the Veridex platform for research and academic purposes. "
        "ESG scores are calculated using a transparent weighting methodology. "
        "Data is based on publicly available and synthetic validated datasets. "
        "This report does not constitute financial or investment advice."
    )

    # Return PDF as bytes for Streamlit download button
    return bytes(pdf.output())