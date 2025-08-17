# app.py
import os
import io
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import requests
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import simpleSplit

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# =========================
#   CONFIG & ENV
# =========================
st.set_page_config(
    page_title="Travel Itinerary Planner",
    page_icon="🗺️",
    layout="wide",
)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing. Please set it in Streamlit Secrets.")
    st.stop()

# Global CSS
st.markdown("""
<style>
:root { --card-bg: #111418; --accent: #ff6a3d; --subtle: #30343a; }
.stApp { background: #0c0f12; color: #e6e6e6; }
.sidebar .sidebar-content { background: #0c0f12; }
.block-container { padding-top: 1.8rem; }
hr { border: none; border-top: 1px solid #23272e; margin: .8rem 0 1.2rem 0; }
.card {
    background: var(--card-bg); border: 1px solid #1f242b; border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 2px 12px rgba(0,0,0,.2);
}
.small { font-size: 0.9rem; opacity: .85; }
.badge { display:inline-block; padding: 4px 10px; border-radius: 999px;
         border:1px solid #2b3139; background:#13171c; margin-right:6px; }
h1, h2, h3 { letter-spacing: .3px; }
button[kind="primary"] { background: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# =========================
#   LLM (Groq)
# =========================
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a meticulous travel assistant. Create a {days}-day itinerary for {city} "
     "based on these interests: {interests}. "
     "Rules:\n"
     "- Output in Markdown with headings '## Day 1', '## Day 2', ...\n"
     "- 4–6 bullet points per day with time hints (e.g., 9:00 AM — Gateway of India)\n"
     "- Keep it realistic (nearby places on the same day)\n"
     "- Include short local food suggestions and commute notes where helpful.\n"),
    ("human", "Plan my trip."),
])

def make_itinerary(city: str, interests: List[str], days: int) -> str:
    msg = PROMPT.format_messages(
        city=city,
        interests=", ".join(interests) if interests else "general sightseeing, food, culture",
        days=days
    )
    return llm.invoke(msg).content

# =========================
#   WEATHER (Open-Meteo, no key)
# =========================
WCODE = {
    0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
    45:"Fog", 48:"Depositing rime fog",
    51:"Light drizzle", 53:"Moderate drizzle", 55:"Dense drizzle",
    56:"Freezing drizzle (light)", 57:"Freezing drizzle (dense)",
    61:"Light rain", 63:"Moderate rain", 65:"Heavy rain",
    66:"Freezing rain (light)", 67:"Freezing rain (heavy)",
    71:"Light snowfall", 73:"Moderate snowfall", 75:"Heavy snowfall",
    77:"Snow grains", 80:"Rain showers (slight)", 81:"Rain showers (moderate)", 82:"Rain showers (violent)",
    85:"Snow showers (slight)", 86:"Snow showers (heavy)",
    95:"Thunderstorm (slight/moderate)", 96:"Thunderstorm w/ hail (slight)", 99:"Thunderstorm w/ hail (heavy)"
}

def geocode(city: str) -> Tuple[float, float, str]:
    """Use Open-Meteo Geocoding API to get lat/lon."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": city, "count": 1}, timeout=10)
    r.raise_for_status()
    j = r.json()
    if not j.get("results"):
        raise ValueError("City not found")
    item = j["results"][0]
    return item["latitude"], item["longitude"], f"{item['name']}, {item.get('country','')}"

def get_weather(city: str, days: int) -> Dict:
    lat, lon, label = geocode(city)
    start = datetime.utcnow().date()
    end = start + timedelta(days=days-1)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
        "start_date": start.isoformat(),
        "end_date": end.isoformat()
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    d = r.json()
    out = {
        "label": label,
        "dates": d["daily"]["time"],
        "tmax": d["daily"]["temperature_2m_max"],
        "tmin": d["daily"]["temperature_2m_min"],
        "pop": d["daily"]["precipitation_probability_max"],
        "wcode": d["daily"]["weathercode"],
    }
    return out

def weather_md(city: str, days: int) -> str:
    try:
        w = get_weather(city, days)
        lines = [f"### 🌤 Weather — {w['label']}"]
        for i, dt in enumerate(w["dates"]):
            desc = WCODE.get(w["wcode"][i], "N/A")
            lines.append(
                f"- **{dt}** — {desc}; **{w['tmin'][i]}–{w['tmax'][i]}°C**, "
                f"rain chance: **{w['pop'][i]}%**"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Weather fetch failed: {e}"

# =========================
#   PDF EXPORT
# =========================
def render_pdf(city: str, days: int, interests_text: str, itinerary_md: str, weather_text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    left, right = 2*cm, (width - 2*cm)
    top = height - 2*cm
    line_h = 14

    def draw_block(title: str, body: str, y: float, font="Helvetica", size=11, bullet=False):
        c.setFont("Helvetica-Bold", 12); c.drawString(left, y, title); y -= 12
        c.setFont(font, size)
        maxw = right - left
        def write(txt, y0):
            nonlocal c
            for ln in txt.split("\n"):
                if bullet and ln.strip().startswith(("*", "-", "•")):
                    ln = "• " + ln.lstrip("*-• ").strip()
                for seg in simpleSplit(ln, font, size, maxw):
                    y0 -= line_h
                    if y0 < 2*cm:
                        c.showPage(); c.setFont(font, size); y0 = top
                    c.drawString(left, y0, seg)
            return y0
        y = write(body, y)
        return y - 8

    # Title
    c.setTitle(f"{city} — {days}-Day Itinerary")
    c.setFont("Helvetica-Bold", 16)
    y = top
    c.drawString(left, y, f"{city.title()} — {days}-Day Itinerary")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    y -= 12

    # Sections
    y = draw_block("Interests", interests_text or "—", y)
    y = draw_block("Weather", weather_text.replace("### ", ""), y)
    # Normalize Markdown headings to plain
    norm = []
    for ln in itinerary_md.splitlines():
        if ln.startswith("### "): ln = ln[4:]
        if ln.startswith("## "): ln = ln[3:]
        if ln.startswith("# "): ln = ln[2:]
        if ln.startswith("# "): ln = ln[1:]
        norm.append(ln)
    y = draw_block("Itinerary", "\n".join(norm), y, bullet=True)

    c.showPage(); c.save()
    buf.seek(0)
    return buf.read()

# =========================
#   UI
# =========================
st.markdown("<h1>🗺️ Travel Itinerary Planner</h1>", unsafe_allow_html=True)
st.write("✨ Plan trips for **1–5 days**, enjoy **live weather updates** (free & instant), and **download a beautiful PDF itinerary** — all in one place.")


with st.sidebar:
    st.markdown("### Trip Inputs")
    city = st.text_input("City", value="")
    interests_csv = st.text_area("Interests (comma-separated)", value="")
    days = st.slider("Days", min_value=1, max_value=5, value=1, step=1)
    st.markdown("<hr/>", unsafe_allow_html=True)
    go = st.button("🚀 Generate Plan", use_container_width=True)

col1, col2 = st.columns([0.58, 0.42])

if go:
    if not city.strip():
        st.sidebar.error("Please fill the City field before generating a plan.")
        with col1:
            st.markdown("#### 📅 Itinerary")
            st.info("Fill inputs in the sidebar and click **Generate Plan**.")
        with col2:
            st.markdown("#### 🌤 Weather")
            st.info("Weather will appear here after generation.")
    else:
        interests = [s.strip() for s in interests_csv.split(",") if s.strip()]
        with st.spinner("Creating itinerary..."):
            itinerary = make_itinerary(city.strip(), interests, days)
        with st.spinner("Fetching weather..."):
            wtext = weather_md(city.strip(), days)

        with col1:
            st.markdown("#### 📅 Itinerary")
            st.markdown(f"<div class='card'>{itinerary}</div>", unsafe_allow_html=True)

        # Only show weather if fetch did not fail
        if not wtext.startswith("⚠️ Weather fetch failed"):
            with col2:
                st.markdown("#### 🌤 Weather")
                st.markdown(f"<div class='card small'>{wtext}</div>", unsafe_allow_html=True)
        else:
            with col2:
                st.markdown("#### 🌤 Weather")
                st.info("Weather data could not be fetched for this city.")

        # PDF
        pdf_bytes = render_pdf(city.strip(), days, ", ".join(interests), itinerary, wtext)
        fname = f"itinerary_{city.strip().replace(' ', '_').lower()}_{days}d.pdf"
        st.download_button(
            "⬇️ Download PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True
        )
else:
    with col1:
        st.markdown("#### 📅 Itinerary")
        st.info("Fill inputs in the sidebar and click **Generate Plan**.")
    with col2:
        st.markdown("#### 🌤 Weather")
        st.info("Weather will appear here after generation.")
