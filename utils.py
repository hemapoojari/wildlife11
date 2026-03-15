import joblib
import streamlit as st


def load_migration_model():
    model = joblib.load("models/migration_model.pkl")
    return model


# ---------- UI HELPERS ----------

def start_card(title: str, padding: int = 20, extra_style: str = ""):
    """Open a translucent glass-style card with an optional title.

    Parameters
    ----------
    title : str
        Text to display as the card header. Centered by default.
    padding : int, optional
        Internal padding (px) of the card. Default is 20.
    extra_style : str, optional
        Additional CSS rules injected into the container.
    """
    style = f"padding: {padding}px; {extra_style}"
    html = f"""
    <div style="
        background: rgba(255,255,255,0.18);
        border-radius: 20px;
        {style}
        backdrop-filter: blur(18px);
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    ">
    <h4 style='text-align:center; color:#0B3D2E;'>{title}</h4>
    """
    st.markdown(html, unsafe_allow_html=True)


def end_card():
    """Close a glass-style card started with :func:`start_card`."""
    st.markdown("</div>", unsafe_allow_html=True)


def analyze_sentiment(text: str) -> str:
    """Analyze ranger report to determine risk level.

    Parameters
    ----------
    text : str
        The ranger report text to analyze.

    Returns
    -------
    str
        Risk level: "High Risk", "Moderate", or "Low Risk".
    """
    text_lower = text.lower()

    # Keywords for risk assessment
    high_risk_keywords = [
        "poaching", "injured", "dying", "dead", "danger", "attack",
        "trap", "snare", "gunshot", "urgent", "critical",
        "immediate", "emergency", "severe"
    ]
    moderate_keywords = [
        "distress", "conflict", "threat", "concern", "warning",
        "population", "decrease", "sick", "starving"
    ]

    # Count keyword matches
    high_count = sum(1 for kw in high_risk_keywords if kw in text_lower)
    moderate_count = sum(1 for kw in moderate_keywords if kw in text_lower)

    if high_count > 0:
        return "High Risk"
    elif moderate_count > 0:
        return "Moderate"
    else:
        return "Low Risk"