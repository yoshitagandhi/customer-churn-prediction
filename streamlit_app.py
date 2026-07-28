from __future__ import annotations

import streamlit as st

from app.components.layout import configure_page
from app.pages.dashboard.dashboard import render_dashboard_page
from app.pages.prediction import render_prediction_page
from app.utils.session import init_session_state

PAGES = {
    "Overview": render_dashboard_page,
    "Churn assessment": render_prediction_page,
}


def _load_styles() -> None:
    """Apply a scoped visual theme without affecting Streamlit's layout."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        [data-testid="stAppViewContainer"] { background:radial-gradient(circle at 92% -10%,rgba(255,181,71,.22),transparent 24rem),radial-gradient(circle at 6% 82%,rgba(50,190,179,.16),transparent 27rem),linear-gradient(135deg,#f7f8fc,#f5f7ff 52%,#fff8ef); }
        [data-testid="stHeader"] { background:rgba(247,248,252,.78); border-bottom:1px solid rgba(219,225,239,.8); } [data-testid="stSidebar"] { background:linear-gradient(180deg,#fcfdff,#f1f4ff); border-right:1px solid #dce3f2; }
        .block-container { max-width:1320px; padding-top:2.15rem; padding-bottom:3rem; } h1,h2,h3 { color:#17233f; font-family:'Space Grotesk','Segoe UI',sans-serif; letter-spacing:-.035em; } h2 { margin-top:.25rem; } p,div,label { font-family:'DM Sans','Segoe UI',sans-serif; }
        .page-hero { position:relative; overflow:hidden; padding:2.3rem 2.5rem; margin:0 0 1.6rem; border:1px solid rgba(207,218,245,.95); border-radius:28px; background:linear-gradient(118deg,#fff,#f4f7ff 58%,#f1edff); box-shadow:0 18px 44px rgba(54,73,128,.12); }
        .page-hero:after { content:''; position:absolute; right:-4rem; bottom:-6rem; width:17rem; height:17rem; border-radius:50%; background:radial-gradient(circle,rgba(255,188,84,.5) 0 2%,transparent 2.5% 100%); background-size:26px 26px; opacity:.7; transform:rotate(-18deg); }
        .page-hero h1 { position:relative; z-index:1; margin:.35rem 0 .55rem; font-size:clamp(2.15rem,4vw,3rem); max-width:780px; } .page-hero p { position:relative; z-index:1; color:#5d6982; font-size:1.05rem; max-width:700px; } .eyebrow { color:#5864c9; font-size:.73rem; font-weight:800; letter-spacing:.14em; }
        .hero-status { position:relative; z-index:1; display:inline-block; margin-top:1rem; padding:.5rem .9rem; border-radius:999px; background:#e8f8f2; color:#187555; font-weight:700; font-size:.82rem; } .hero-status span { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.4rem; background:#27b58b; box-shadow:0 0 0 4px rgba(39,181,139,.14); animation:pulse 2s infinite; } @keyframes pulse { 50% { box-shadow:0 0 0 7px rgba(39,181,139,0); } }
        .brand-lockup { display:flex; align-items:center; gap:.8rem; margin:.35rem .2rem 1.7rem; } .brand-mark { display:grid; place-items:center; width:46px; height:46px; border-radius:15px; background:linear-gradient(135deg,#172b59,#4354b9 52%,#7d68d9); color:#fff; font-family:'Space Grotesk',sans-serif; font-size:1.08rem; font-weight:700; letter-spacing:-.1em; box-shadow:0 10px 20px rgba(52,72,163,.28); transform:rotate(-5deg); } .brand-mark em { color:#ffd476; font-style:normal; font-size:.85em; } .brand-lockup strong { color:#17233f; font-family:'Space Grotesk',sans-serif; font-size:1.12rem; }.brand-lockup span { display:block; color:#71809a; font-size:.76rem; }
        .workspace-status { margin:.8rem 0 1.25rem; padding:.85rem .9rem; border-radius:16px; background:linear-gradient(135deg,#172b59,#364aa4); color:#eef3ff; box-shadow:0 10px 22px rgba(39,54,112,.18); } .workspace-status b { display:block; color:#fff; font-size:.78rem; letter-spacing:.09em; }.workspace-status span { color:#bfeee6; font-size:.76rem; }
        div[data-testid="stMetric"] { background:rgba(255,255,255,.96); border:1px solid #dde5f3; border-radius:18px; padding:1.05rem; box-shadow:0 10px 26px rgba(42,65,120,.08); border-top:4px solid #6f7cea; transition:transform .18s ease,box-shadow .18s ease; } div[data-testid="stMetric"]:hover { transform:translateY(-3px); box-shadow:0 15px 29px rgba(42,65,120,.14); } [data-testid="stMetricLabel"] { color:#68758e; font-weight:600; } [data-testid="stMetricValue"] { color:#17233f; font-family:'Space Grotesk',sans-serif; }
        [data-testid="stForm"],[data-testid="stExpander"],[data-testid="stVerticalBlockBorderWrapper"] { border:1px solid #cbd8ef; border-radius:20px; background:#ffffff!important; box-shadow:0 12px 28px rgba(42,65,120,.12); } [data-testid="stForm"] { padding:.45rem .8rem .8rem; } [data-testid="stForm"] h4 { color:#4d5d7a; letter-spacing:.02em; } [data-testid="stVerticalBlockBorderWrapper"] h1,[data-testid="stVerticalBlockBorderWrapper"] h2,[data-testid="stVerticalBlockBorderWrapper"] h3,[data-testid="stVerticalBlockBorderWrapper"] p,[data-testid="stVerticalBlockBorderWrapper"] li { color:#17233f!important; }
        .stButton button,.stDownloadButton button,[data-testid="stFormSubmitButton"] button { min-height:2.65rem; border-radius:12px; border:0; background:linear-gradient(100deg,#2eacb0,#5364dc 58%,#835fd3); color:#fff; font-weight:700; transition:transform .18s ease,box-shadow .18s ease; } .stButton button:hover,.stDownloadButton button:hover { filter:brightness(.99); transform:translateY(-2px); box-shadow:0 9px 20px rgba(75,91,205,.28); }
        [data-testid="stDataFrame"] { border:1px solid #dfe6f4; border-radius:14px; overflow:hidden; } [data-testid="stRadio"] label { border-radius:10px; padding:.4rem .5rem; transition:background .15s; } [data-testid="stRadio"] label:hover { background:#eaf0ff; } .sidebar-footer { margin-top:1.8rem; padding-top:1rem; border-top:1px solid #dfe6f4; color:#73819a; font-size:.76rem; line-height:1.55; }
        .attention-heading { display:flex; align-items:center; gap:.65rem; margin:1.7rem 0 .8rem; color:#17233f; font-family:'Space Grotesk',sans-serif; font-size:1.35rem; font-weight:700; } .attention-heading i { display:grid; place-items:center; width:32px; height:32px; border-radius:10px; background:#e7ebff; color:#5364dc; font-style:normal; font-size:1rem; }
        .section-heading { display:flex; align-items:center; width:fit-content; margin:1.65rem 0 1rem; padding:.72rem 1.05rem; border:1px solid #168b86; border-left:5px solid #9ee7dc; border-radius:12px; background:linear-gradient(105deg,#126d6c,#229b91); color:#ffffff!important; font-family:'Space Grotesk','Segoe UI',sans-serif!important; font-size:1.45rem; font-weight:700; line-height:1.2; box-shadow:0 10px 22px rgba(18,109,108,.24); }
        .business-insight-card { margin:.25rem 0 1.15rem; padding:1.45rem 1.55rem; border:1px solid #a9ddd7; border-left:6px solid #15958b; border-radius:18px; background:linear-gradient(120deg,#ffffff,#f2fcfa); box-shadow:0 14px 30px rgba(15,118,110,.14); } .business-insight-card h3 { margin:0 0 .72rem; color:#103f49!important; font-family:'Space Grotesk','Segoe UI',sans-serif; font-size:1.55rem; } .business-insight-card p { margin:0; color:#244c58!important; font-size:1.04rem; font-weight:600; line-height:1.6; } .insight-recommendations { margin-top:1.1rem; padding-top:.95rem; border-top:1px solid #ccebe6; color:#123f49; } .insight-recommendations strong { color:#0f766e; font-size:.92rem; letter-spacing:.02em; } .insight-recommendations ul { margin:.55rem 0 0; padding-left:1.25rem; } .insight-recommendations li { margin:.45rem 0; color:#244c58!important; font-weight:600; }
        .status-spotlight { position:relative; overflow:hidden; padding:1.25rem 1.35rem; border-radius:20px; color:#fff; box-shadow:0 14px 28px rgba(28,45,107,.2); } .status-spotlight:after { content:''; position:absolute; width:9rem; height:9rem; right:-3rem; top:-4rem; border:1px solid rgba(255,255,255,.25); border-radius:50%; box-shadow:0 0 0 22px rgba(255,255,255,.07),0 0 0 44px rgba(255,255,255,.04); } .status-spotlight.healthy { background:linear-gradient(120deg,#157d72,#27aa93); } .status-spotlight.stable { background:linear-gradient(120deg,#4454bc,#7367d7); } .status-spotlight.attention { background:linear-gradient(120deg,#bf553e,#e98a3d); } .spotlight-kicker { position:relative; z-index:1; color:rgba(255,255,255,.75); font-size:.7rem; font-weight:800; letter-spacing:.13em; } .spotlight-title { position:relative; z-index:1; margin:.3rem 0; font-family:'Space Grotesk',sans-serif; font-size:1.65rem; font-weight:700; } .spotlight-copy { position:relative; z-index:1; margin:0; max-width:36rem; color:rgba(255,255,255,.9)!important; font-size:.93rem!important; } .spotlight-score { position:absolute; z-index:1; right:1.3rem; bottom:1.1rem; font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; }
        .health-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.7rem; } .health-check { padding:.85rem .9rem; border:1px solid #dbe9e5; border-radius:14px; background:#fbfffe; color:#2c5e57; font-size:.87rem; font-weight:600; } .health-check b { display:block; margin-bottom:.18rem; color:#168069; font-size:.74rem; letter-spacing:.06em; } .health-check.offline { border-color:#dce3ef; background:#f8fafc; color:#55637a; } .health-check.offline b { color:#71809a; }
        .explore-panel { padding:1.25rem 1.35rem 1.4rem; border:1px solid #dce3f4; border-radius:22px; background:linear-gradient(135deg,rgba(255,255,255,.9),rgba(239,243,255,.9)); box-shadow:0 12px 26px rgba(47,65,126,.08); } .explore-panel > p { margin-top:-.35rem; } .explore-card { min-height:110px; margin-top:.55rem; padding:1rem; border:1px solid #dfe6f5; border-radius:15px; background:#fff; transition:transform .18s ease,box-shadow .18s ease; } .explore-card:hover { transform:translateY(-3px); box-shadow:0 12px 22px rgba(52,70,130,.13); } .explore-card h3 { margin:0 0 .3rem; font-size:1.03rem; } .explore-card p { margin:0 0 .65rem; color:#65728a; font-size:.86rem; } .explore-card span { color:#5864c9; font-size:.74rem; font-weight:800; letter-spacing:.06em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark">C<em>•</em>P</div>
            <div><strong>ChurnPulse</strong><span>Retention intelligence</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption("MODEL WORKSPACE")
    st.sidebar.markdown(
        "<div class='workspace-status'><b>● LIVE MODEL</b><span>Retention signal online</span></div>",
        unsafe_allow_html=True,
    )
    selected = st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.sidebar.markdown("<div class='sidebar-footer'>IBM Telco Customer Churn<br>Decision-support demo</div>", unsafe_allow_html=True)
    return selected


def main() -> None:
    configure_page(title="ChurnPulse | Customer Retention Intelligence", icon="CP")
    init_session_state()
    _load_styles()
    selected_page = _render_sidebar()
    PAGES[selected_page]()


if __name__ == "__main__":
    main()
