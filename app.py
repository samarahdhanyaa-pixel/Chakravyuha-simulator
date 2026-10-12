"""CHAKRAVYUHA — an AI decision-making simulator under incomplete information."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from agent import STRATEGIES
from decision_engine import format_action, legal_actions
from experiments import csv_bytes, run_experiment, tournament
from integration import advance, current_observation, new_simulation, pending_decision, run_to_termination, scan
from reward import reward_breakdown
from scenario_generator import SCENARIOS
from state import DIRECTIONS
from visualization import battlefield_html, comparison_chart, legend_html, uncertainty_radar

PAGE_NAMES = ("Command Centre", "Strategy Lab", "Experiment Studio", "Architecture", "Research Notes")
SCENARIO_LABELS = {
    "The Outer Gate": "The Outer Gate · Easy / High visibility",
    "Veil of Ash": "Veil of Ash · Medium / Partial visibility",
    "The Silent Wheel": "The Silent Wheel · Hard / Severe uncertainty",
}

st.set_page_config(
    page_title="CHAKRAVYUHA · Intelligence of War",
    page_icon="✹",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles(presentation: bool = False, entered: bool = True) -> None:
    st.markdown(
        f"""
        <style>
        :root {{
          --obsidian:#0a0909; --charcoal:#171310; --gold:#c6a15b; --pale:#e9d4a0;
          --parchment:#e8dcc4; --crimson:#9c3434; --bronze:#6f5937; --muted:#a99d86;
        }}
        html, body, [data-testid="stAppViewContainer"], .stApp {{
          background:
            radial-gradient(ellipse at 52% 0%, rgba(116,77,30,.14), transparent 40%),
            radial-gradient(ellipse at 8% 78%, rgba(156,52,52,.07), transparent 32%),
            #0a0909 !important;
          color:var(--parchment);
        }}
        [data-testid="stHeader"] {{ background:rgba(10,9,9,.55); }}
        [data-testid="stToolbar"], #MainMenu, footer {{ visibility:hidden; }}
        .block-container {{ max-width:1560px; padding-top:1.65rem; padding-bottom:3rem; }}
        [data-testid="stSidebar"] {{
          background:linear-gradient(180deg,rgba(23,19,16,.99),rgba(12,10,9,.99));
          border-right:1px solid rgba(198,161,91,.2);
        }}
        [data-testid="stSidebar"] * {{ color:var(--parchment); }}
        h1,h2,h3,h4 {{ font-family:Georgia,'Times New Roman',serif !important; color:var(--pale) !important; }}
        p, label, span, div {{ font-family:Inter,ui-sans-serif,system-ui,sans-serif; }}
        .stCaption, [data-testid="stCaptionContainer"] {{ color:#a99d86 !important; }}
        [data-testid="stMetric"] {{
          background:linear-gradient(145deg,rgba(28,23,18,.92),rgba(15,13,12,.92));
          border:1px solid rgba(198,161,91,.25); border-radius:12px; padding:13px 16px;
          min-height:95px;
        }}
        [data-testid="stMetricLabel"] {{ color:#b6a989 !important; font-size:.75rem !important; letter-spacing:.11em; text-transform:uppercase; }}
        [data-testid="stMetricValue"] {{ color:#ead7a4 !important; font-family:Georgia,serif; }}
        [data-testid="stMetricDelta"] {{ color:#d8c698 !important; }}
        .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {{
          color:#160f08 !important; background:linear-gradient(135deg,#e1c579,#b88d42) !important;
          border:1px solid #e9d4a0 !important; border-radius:8px !important;
          font-weight:700 !important; letter-spacing:.06em; transition:transform .18s,filter .18s,box-shadow .18s;
          box-shadow:0 5px 18px rgba(198,161,91,.12);
        }}
        .stButton button:hover, .stDownloadButton button:hover {{
          transform:translateY(-1px); filter:brightness(1.08);
          box-shadow:0 8px 23px rgba(198,161,91,.23);
        }}
        [data-testid="stBaseButton-secondary"] {{
          color:#e5d9c0 !important; background:#1b1713 !important; border-color:rgba(198,161,91,.36) !important;
        }}
        input, textarea, [data-baseweb="select"] > div, [data-testid="stNumberInput"] {{
          background:#171310 !important; border-color:rgba(198,161,91,.28) !important; color:#e8dcc4 !important;
        }}
        [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {{ color:#d6cbb2 !important; }}
        [data-testid="stHorizontalBlock"] {{ gap:1rem; }}
        [data-testid="stPlotlyChart"] {{ border:1px solid rgba(198,161,91,.16); border-radius:12px; background:rgba(21,18,15,.7); padding:6px; }}
        [data-testid="stDataFrame"] {{ border:1px solid rgba(198,161,91,.2); border-radius:10px; }}
        [data-testid="stExpander"] {{ border:1px solid rgba(198,161,91,.18) !important; background:rgba(23,19,16,.56); border-radius:10px; }}
        .eyebrow {{ color:#c6a15b; font-size:.68rem; letter-spacing:.22em; text-transform:uppercase; font-weight:700; }}
        .page-title {{ font:clamp(1.8rem,3.1vw,2.65rem) Georgia,serif; color:#ead4a0; letter-spacing:.035em; }}
        .subtitle {{ color:#b4a78f; font-size:.92rem; line-height:1.6; }}
        .topbar {{ display:flex; align-items:center; justify-content:space-between; padding:0 0 17px; margin-bottom:14px; border-bottom:1px solid rgba(198,161,91,.26); }}
        .brand-lockup {{ display:flex; align-items:center; gap:13px; }}
        .brand-seal {{ width:42px; height:42px; display:grid; place-items:center; border:1px solid #9c783e; border-radius:50%; color:#e4c47b; font:24px Georgia,serif; box-shadow:inset 0 0 13px rgba(198,161,91,.12); }}
        .brand-word {{ color:#ebdbb2; font:1.02rem Georgia,serif; letter-spacing:.17em; }}
        .brand-sub {{ color:#94866f; font-size:.67rem; letter-spacing:.16em; text-transform:uppercase; }}
        .status-pill {{ display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border:1px solid rgba(198,161,91,.28); border-radius:20px; background:rgba(198,161,91,.07); color:#d5c190; font-size:.75rem; letter-spacing:.08em; text-transform:uppercase; }}
        .status-dot {{ width:7px; height:7px; background:#9c3434; border-radius:50%; box-shadow:0 0 10px #9c3434; }}
        .panel-heading {{ display:flex; justify-content:space-between; align-items:baseline; margin:8px 0 12px; }}
        .panel-heading h3 {{ margin:0; font-size:1.05rem; }}
        .panel-kicker {{ color:#8c7a5c; font-size:.66rem; letter-spacing:.13em; text-transform:uppercase; }}
        .insight-box {{ border:1px solid rgba(198,161,91,.25); border-left:3px solid #c6a15b; border-radius:8px; padding:14px 16px; background:linear-gradient(100deg,rgba(198,161,91,.085),rgba(23,19,16,.78)); color:#d4c7ad; line-height:1.55; }}
        .parchment-note {{ background:linear-gradient(140deg,#e8dcc4,#d4c3a0); color:#251c12; border-radius:12px; padding:18px 20px; box-shadow:0 12px 36px rgba(0,0,0,.2); }}
        .parchment-note h4 {{ color:#332313 !important; margin:0 0 9px; }}
        .parchment-note p,.parchment-note li {{ color:#463723; line-height:1.6; font-size:.88rem; }}
        .event-list {{ display:flex; flex-direction:column; gap:7px; }}
        .event-row {{ display:grid; grid-template-columns:58px 1fr auto; gap:10px; align-items:center; border:1px solid rgba(198,161,91,.14); background:rgba(25,21,17,.74); border-radius:7px; padding:9px 11px; font-size:.78rem; }}
        .event-step {{ color:#b99a58; font-family:Georgia,serif; }}
        .event-text {{ color:#d5cbb6; }}
        .event-reward {{ color:#bfae8a; font-variant-numeric:tabular-nums; }}
        .legend {{ display:flex; flex-wrap:wrap; gap:8px 14px; margin:12px 0 4px; color:#b7aa91; font-size:.72rem; }}
        .legend span {{ display:inline-flex; align-items:center; gap:5px; }}
        .legend i {{ width:18px; height:18px; display:inline-grid; place-items:center; border:1px solid rgba(198,161,91,.24); border-radius:4px; font-style:normal; }}
        .legend-agent {{ color:#1c1308; background:#dfbd67; }} .legend-goal {{ color:#ead590; background:#46351a; }}
        .legend-enemy {{ color:#ffc2b3; background:#512222; }} .legend-hazard {{ color:#eac783; background:#473619; }}
        .legend-wall {{ color:#bdad92; background:#443c31; }} .legend-fog {{ color:#827a6d; background:#171514; }}
        .legend-visited {{ color:#e9d4a0; background:#30291f; }}
        .board-stage {{ position:relative; overflow:hidden; padding:clamp(7px,1.1vw,14px); border:1px solid rgba(198,161,91,.27); border-radius:13px; background:radial-gradient(ellipse at 50% 48%,rgba(94,67,32,.18),rgba(15,13,11,.97) 73%); }}
        .board-stage:before,.board-stage:after {{ content:''; position:absolute; z-index:0; left:50%; top:50%; transform:translate(-50%,-50%); border:1px solid rgba(198,161,91,.065); border-radius:50%; pointer-events:none; }}
        .board-stage:before {{ width:82%; aspect-ratio:1; }} .board-stage:after {{ width:56%; aspect-ratio:1; }}
        .battle-grid {{ position:relative; z-index:1; width:min(100%,610px); margin:auto; display:grid; gap:3px; }}
        .tile {{ display:flex; align-items:center; justify-content:center; aspect-ratio:1; min-width:0; border-radius:4px; font-size:clamp(.62rem,1.35vw,1rem); border:1px solid rgba(198,161,91,.08); transition:transform .15s,box-shadow .18s,background .18s; user-select:none; }}
        .tile:hover {{ transform:scale(1.08); z-index:2; box-shadow:0 0 13px rgba(224,189,103,.23); }}
        .tile.fog {{ color:#746c60; background:linear-gradient(145deg,rgba(30,28,25,.95),rgba(20,19,18,.92)); border-color:rgba(151,140,118,.06); }}
        .tile.known {{ color:#aa9c81; background:rgba(63,53,38,.6); }}
        .tile.visited {{ background:rgba(198,161,91,.11); color:#c5a65f; }}
        .tile.wall {{ color:#c6b79d; background:linear-gradient(145deg,#4e4437,#2e2923); border-color:rgba(183,157,109,.3); }}
        .tile.hazard {{ color:#f1cb70; background:radial-gradient(circle,#655022,#3d321d); box-shadow:inset 0 0 8px rgba(230,191,91,.16); }}
        .tile.enemy {{ color:#ffd0c2; background:radial-gradient(circle,#7d3530,#42201e); box-shadow:inset 0 0 9px rgba(214,71,58,.23); }}
        .tile.goal {{ color:#fae39a; background:radial-gradient(circle,#79612f,#382b18); border-color:#d2b369; box-shadow:0 0 11px rgba(215,179,91,.22); }}
        .tile.agent {{ color:#241809; background:radial-gradient(circle,#f4db92,#c2933e); border-color:#f7e4aa; box-shadow:0 0 15px rgba(235,199,104,.6); font-weight:800; }}
        .tile.next {{ outline:2px solid #e8cd80; outline-offset:1px; }}
        .tile.truth-hidden {{ border-style:dashed; }}
        .score-row {{ display:grid; grid-template-columns:30px 1fr 65px; align-items:center; gap:9px; margin:8px 0; color:#cec2a8; font-size:.78rem; }}
        .score-bar {{ height:5px; background:#28221b; border-radius:9px; overflow:hidden; }}
        .score-fill {{ height:100%; background:linear-gradient(90deg,#73592f,#e3c36f); border-radius:9px; }}
        .pipeline {{ display:grid; grid-template-columns:repeat(6,1fr); gap:5px; margin:14px 0; }}
        .pipeline-step {{ position:relative; padding:11px 8px; border:1px solid rgba(198,161,91,.22); border-radius:7px; text-align:center; background:rgba(25,21,17,.78); color:#d5c29a; font-size:.7rem; }}
        .pipeline-step strong {{ display:block; color:#ead4a0; font: .82rem Georgia,serif; margin-bottom:5px; }}
        .pipeline-arrow {{ color:#927541; }}
        .hero-wrap {{ position:relative; min-height:80vh; overflow:hidden; display:grid; grid-template-columns:1.2fr .8fr; align-items:center; gap:4vw; padding:clamp(28px,6vw,80px); border:1px solid rgba(198,161,91,.29); border-radius:20px; background:radial-gradient(ellipse at 79% 46%,rgba(143,85,31,.18),transparent 37%),linear-gradient(120deg,rgba(13,11,10,.99),rgba(27,21,15,.89)); box-shadow:0 35px 95px rgba(0,0,0,.36); }}
        .hero-wrap:before {{ content:''; position:absolute; inset:10px; border:1px solid rgba(198,161,91,.13); border-radius:15px; pointer-events:none; }}
        .hero-copy,.hero-art {{ position:relative; z-index:1; }}
        .hero-title {{ font:clamp(2.7rem,7.3vw,6.2rem)/.96 Georgia,'Times New Roman',serif; letter-spacing:.08em; color:#ebd69d; text-shadow:0 4px 38px rgba(218,177,86,.17); margin:18px 0; }}
        .hero-question {{ color:#e4d6b6; font:clamp(1.05rem,1.8vw,1.45rem)/1.45 Georgia,serif; max-width:640px; }}
        .hero-tagline {{ color:#a29378; font-size:.9rem; letter-spacing:.04em; margin:24px 0 27px; }}
        .hero-meta {{ display:flex; gap:9px; flex-wrap:wrap; margin-top:35px; }}
        .meta-chip {{ color:#baa578; border:1px solid rgba(198,161,91,.2); background:rgba(198,161,91,.05); border-radius:20px; padding:7px 11px; font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; }}
        .mandala {{ width:min(100%,470px); margin:auto; filter:drop-shadow(0 0 32px rgba(198,161,91,.16)); animation:slowturn 100s linear infinite; }}
        @keyframes slowturn {{ to {{ transform:rotate(360deg); }} }}
        .mandala-ring {{ fill:none; stroke:rgba(211,174,98,.52); stroke-width:1; }}
        .mandala-petal {{ fill:rgba(198,161,91,.035); stroke:rgba(211,174,98,.42); stroke-width:1; }}
        .mandala-center {{ fill:#ba9049; filter:drop-shadow(0 0 13px rgba(230,193,102,.7)); }}
        [data-testid="stRadio"] > div {{ gap:.3rem; }}
        [data-testid="stRadio"] label p {{ font-size:.82rem; }}
        .stTabs [data-baseweb="tab-list"] {{ gap:8px; background:transparent; }}
        @media (max-width:900px) {{
          .hero-wrap {{ grid-template-columns:1fr; min-height:auto; }}
          .hero-art {{ max-width:300px; margin:auto; }}
          .pipeline {{ grid-template-columns:repeat(3,1fr); }}
        }}
        @media (max-width:650px) {{
          .block-container {{ padding-left:1rem; padding-right:1rem; }}
          .pipeline {{ grid-template-columns:repeat(2,1fr); }}
          .event-row {{ grid-template-columns:46px 1fr auto; }}
          .battle-grid {{ gap:2px; }}
        }}
        {'[data-testid="stSidebar"]{display:none;} .block-container{max-width:1700px;}' if presentation else ''}
        {'[data-testid="stSidebar"]{display:none;}' if not entered else ''}
        </style>
        """,
        unsafe_allow_html=True,
    )


def mandala_svg() -> str:
    petals = []
    for ring, count, radius in ((0, 8, 65), (1, 16, 110), (2, 24, 155)):
        for index in range(count):
            angle = 360 * index / count
            petals.append(
                f'<ellipse class="mandala-petal" cx="200" cy="{200-radius}" rx="{8 if ring == 0 else 6}" ry="{28 if ring == 0 else 22}" transform="rotate({angle} 200 200)" />'
            )
    rings = "".join(
        f'<circle class="mandala-ring" cx="200" cy="200" r="{radius}" />'
        for radius in (24, 48, 82, 125, 168, 190)
    )
    return (
        '<svg class="mandala" viewBox="0 0 400 400" role="img" aria-label="Chakravyuha radial formation">'
        + rings
        + "".join(petals)
        + '<path class="mandala-ring" stroke-dasharray="2 8" stroke-width="2" d="M200 12V388M12 200H388M67 67L333 333M333 67L67 333" />'
        + '<circle class="mandala-center" cx="200" cy="200" r="6" /></svg>'
    )


def initialize_state() -> None:
    st.session_state.setdefault("entered", False)
    st.session_state.setdefault("presentation", False)
    st.session_state.setdefault("instructor_truth", False)
    st.session_state.setdefault("page", PAGE_NAMES[0])
    st.session_state.setdefault(
        "app_config",
        {
            "scenario": "The Outer Gate",
            "difficulty": "Easy",
            "visibility": "High",
            "strategy": "Uncertainty-Aware AI",
            "seed": 17,
            "start": (12, 1),
            "goal": (6, 6),
        },
    )
    if "simulation" not in st.session_state:
        cfg = st.session_state.app_config
        st.session_state.simulation = new_simulation(
            strategy=cfg["strategy"],
            difficulty=cfg["difficulty"],
            visibility=cfg["visibility"],
            seed=cfg["seed"],
        )
    st.session_state.setdefault("experiment_result", None)


def sidebar_controls() -> None:
    sim = st.session_state.simulation
    cfg = st.session_state.app_config
    with st.sidebar:
        st.markdown(
            '<div class="brand-lockup" style="margin:9px 0 22px"><div class="brand-seal">✹</div>'
            '<div><div class="brand-word">CHAKRAVYUHA</div><div class="brand-sub">Field command</div></div></div>',
            unsafe_allow_html=True,
        )
        preset_labels = list(SCENARIO_LABELS.values())
        custom_label = "Seeded generator · custom"
        selected_label = SCENARIO_LABELS.get(cfg["scenario"], custom_label)
        selected_label = selected_label if selected_label in preset_labels else custom_label
        with st.form("deployment_configuration"):
            chosen_label = st.selectbox("Battlefield briefing", preset_labels + [custom_label], index=(preset_labels + [custom_label]).index(selected_label))
            difficulty_index = ("Easy", "Medium", "Hard").index(cfg["difficulty"])
            difficulty = st.selectbox("Difficulty", ("Easy", "Medium", "Hard"), index=difficulty_index)
            visibility_index = ("High", "Partial", "Severe").index(cfg["visibility"])
            visibility = st.selectbox("Visibility", ("High", "Partial", "Severe"), index=visibility_index)
            strategy_index = STRATEGIES.index(cfg["strategy"])
            strategy = st.selectbox("Active policy", STRATEGIES, index=strategy_index)
            seed = st.number_input("Custom seed", min_value=0, max_value=2_147_483_647, value=int(cfg["seed"]), step=1)
            submitted = st.form_submit_button("Deploy configuration", width="stretch")
        if submitted:
            chosen_scenario = next((key for key, label in SCENARIO_LABELS.items() if label == chosen_label), None)
            if chosen_scenario:
                scenario = SCENARIOS[chosen_scenario]
                config = {
                    "scenario": chosen_scenario,
                    "difficulty": scenario["difficulty"],
                    "visibility": scenario["visibility"],
                    "strategy": strategy,
                    "seed": scenario["seed"],
                    "start": scenario["start"],
                    "goal": scenario["goal"],
                }
            else:
                config = {
                    "scenario": "Seeded generator",
                    "difficulty": difficulty,
                    "visibility": visibility,
                    "strategy": strategy,
                    "seed": int(seed),
                }
            st.session_state.app_config = config
            st.session_state.simulation = new_simulation(
                strategy=config["strategy"],
                difficulty=config["difficulty"],
                visibility=config["visibility"],
                seed=config["seed"],
                start=config.get("start"),
                goal=config.get("goal"),
            )
            st.session_state.replay_index = 0
            st.session_state.pop("replay_slider", None)
            st.rerun()

        st.markdown("---")
        st.toggle("Instructor truth layer", key="instructor_truth", help="Displays the full map to the viewer. It is not passed to the agent.")
        st.toggle("Presentation mode", key="presentation", help="Hides the control rail for a larger demonstration view.")
        st.markdown(
            '<div class="parchment-note" style="margin-top:16px"><h4>Mission brief</h4>'
            '<p>Enter the inner ring using partial observations. Unknown tiles stay hidden from the policy until discovered.</p>'
            '<p><b>Objective:</b> reach the central command point.<br><b>Protocol:</b> one move per turn; scans reveal further terrain.</p></div>',
            unsafe_allow_html=True,
        )
        st.caption("LOCAL SIMULATION · NO EXTERNAL AI SERVICE")


def show_landing() -> None:
    st.markdown(
        '<div class="hero-wrap"><div class="hero-copy">'
        '<div class="eyebrow">An AI decision-making simulator under incomplete information</div>'
        '<div class="hero-title">CHAKRA<br>VYUHA</div>'
        '<div class="hero-question">Can intelligence find a path when truth is incomplete?</div>'
        '<div class="hero-tagline">“Every move has a consequence. Every unknown hides a possibility.”</div>'
        '<p class="subtitle" style="max-width:590px">A live strategic experiment: observe a veiled battlefield, estimate exposure, choose a move, and compare the outcome against the hidden truth.</p>'
        '</div><div class="hero-art">'
        + mandala_svg()
        + '</div><div style="grid-column:1/-1;position:relative;z-index:2">'
        + '<div class="eyebrow" style="margin-bottom:12px">Decision · uncertainty · consequence</div>'
        + "</div></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns([1.25, 2.2, 1.3])
    with cols[0]:
        if st.button("ENTER THE BATTLEFIELD", width="stretch", type="primary"):
            st.session_state.entered = True
            cfg = st.session_state.app_config
            st.session_state.simulation = new_simulation(
                strategy=cfg["strategy"],
                difficulty=cfg["difficulty"],
                visibility=cfg["visibility"],
                seed=cfg["seed"],
                start=cfg.get("start"),
                goal=cfg.get("goal"),
            )
            st.session_state.replay_index = 0
            st.session_state.pop("replay_slider", None)
            st.rerun()
    with cols[1]:
        st.markdown(
            '<div style="padding:12px 0;color:#988b73;font-size:.78rem;line-height:1.6">'
            'Three policies share one environment. The evaluator sees ground truth; each agent receives only explored terrain and the known objective.</div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            '<div class="hero-meta"><span class="meta-chip">Python · Streamlit</span><span class="meta-chip">Seeded simulation</span><span class="meta-chip">Fog of war</span></div>',
            unsafe_allow_html=True,
        )


def app_header(sim: dict[str, Any]) -> None:
    world = sim["world"]
    status = {"running": "In the field", "success": "Objective secured", "failure": "Mission intercepted", "timeout": "Time limit reached"}.get(world.status, world.status)
    dot_color = "#85a77e" if world.status == "success" else ("#9c3434" if world.status == "failure" else "#c6a15b")
    st.markdown(
        f'<div class="topbar"><div class="brand-lockup"><div class="brand-seal">✹</div>'
        f'<div><div class="brand-word">CHAKRAVYUHA</div><div class="brand-sub">{html.escape(world.difficulty)} · {html.escape(sim["visibility"])} visibility · seed {world.seed}</div></div></div>'
        f'<div class="status-pill"><i class="status-dot" style="background:{dot_color}"></i>{html.escape(status)}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pipeline">'
        '<div class="pipeline-step"><strong>01 · Environment</strong>Hidden battlefield state</div>'
        '<div class="pipeline-step"><strong>02 · Observation</strong>Explored cells only</div>'
        '<div class="pipeline-step"><strong>03 · Policy</strong>Score legal moves</div>'
        '<div class="pipeline-step"><strong>04 · Action</strong>Move or scan</div>'
        '<div class="pipeline-step"><strong>05 · Update</strong>Reward + discoveries</div>'
        '<div class="pipeline-step"><strong>06 · Evaluate</strong>Episode metrics</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_navigation() -> str:
    selected = st.radio(
        "Application sections",
        PAGE_NAMES,
        horizontal=True,
        label_visibility="collapsed",
        key="page",
    )
    return selected


def _score_panel(decision: Any) -> None:
    if decision is None:
        st.info("No move is available because the episode has ended.")
        return
    st.markdown(
        f'<div class="eyebrow">Next recommended move</div><div style="font:1.8rem Georgia;color:#f1dda5;margin:7px 0 4px">{format_action(decision.action)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="insight-box">{html.escape(decision.explanation)}</div>', unsafe_allow_html=True)
    st.markdown("**Candidate action scores**" if decision.scores else "")
    values = list(decision.scores.values())
    span = max(1.0, max(values) - min(values)) if values else 1.0
    for action, score in decision.scores.items():
        width = 18 + 82 * ((score - min(values)) / span) if values else 0
        st.markdown(
            f'<div class="score-row"><span>{html.escape(format_action(action))}</span>'
            f'<div class="score-bar"><div class="score-fill" style="width:{width:.1f}%"></div></div>'
            f'<span style="text-align:right">{score:.2f}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption(f"Exposure index: {decision.risk_index:.0f}/100 · uncertainty index: {decision.uncertainty_index:.0f}/100. Indices are heuristic measures, not calibrated probabilities.")
    with st.expander("Facts available to this decision"):
        for fact in decision.known_facts:
            st.markdown(f"- {html.escape(fact)}")


def render_command_centre() -> None:
    sim = st.session_state.simulation
    world = sim["world"]
    observation = current_observation(sim)
    decision = pending_decision(sim)
    max_steps = {"Easy": 90, "Medium": 80, "Hard": 72}[world.difficulty]
    st.markdown('<div class="eyebrow">Operations · current episode</div><div class="page-title">Command Centre</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{html.escape(SCENARIOS.get(st.session_state.app_config["scenario"], {}).get("description", "A generated maze of unknown terrain, hidden hazards, and shifting strategic choices."))}</p>', unsafe_allow_html=True)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Simulation step", f"{world.step} / {max_steps}")
    metric_cols[1].metric("Cumulative reward", f"{world.cumulative_reward:.1f}")
    metric_cols[2].metric("Known threats", f"{len(observation.visible_hazards) + len(observation.visible_enemies)}")
    metric_cols[3].metric("Unknown cells", f"{world.rows * world.cols - len(observation.visible_cells)}")
    metric_cols[4].metric("Objective", {"running": "In progress", "success": "Secured", "failure": "Failed", "timeout": "Timed out"}[world.status])

    battlefield_col, intel_col = st.columns([1.55, 0.85], gap="large")
    with battlefield_col:
        st.markdown('<div class="panel-heading"><h3>Battlefield · Chakravyuha formation</h3><span class="panel-kicker">13 × 13 · explored terrain persists</span></div>', unsafe_allow_html=True)
        history_index = st.session_state.get("replay_index", len(sim["history"]) - 1)
        history_index = min(history_index, len(sim["history"]) - 1)
        snapshot = sim["history"][history_index]
        truth_mode = st.session_state.instructor_truth
        if truth_mode:
            st.warning("INSTRUCTOR INFORMATION — complete ground truth is visible here only. The agent still receives its partial observation.", icon="🚩")
        st.markdown(battlefield_html(snapshot, instructor_truth=truth_mode), unsafe_allow_html=True)
        st.markdown(legend_html(), unsafe_allow_html=True)
        st.caption("The objective is known from the mission briefing. Cell contents remain veiled until observed.")

        st.markdown('<div class="panel-heading" style="margin-top:22px"><h3>Episode controls</h3><span class="panel-kicker">step · scan · replay</span></div>', unsafe_allow_html=True)
        control_cols = st.columns([1, 1, 1, 1])
        if control_cols[0].button("Advance one move", disabled=world.status != "running", width="stretch"):
            advance(sim)
            st.session_state.replay_index = len(sim["history"]) - 1
            st.session_state.replay_slider = st.session_state.replay_index
            st.rerun()
        if control_cols[1].button("Scan formation", disabled=world.status != "running", width="stretch", help="Spend a turn to reveal additional nearby terrain."):
            scan(sim)
            st.session_state.replay_index = len(sim["history"]) - 1
            st.session_state.replay_slider = st.session_state.replay_index
            st.rerun()
        if control_cols[2].button("Run to outcome", disabled=world.status != "running", width="stretch"):
            with st.spinner("Executing the seeded episode…"):
                run_to_termination(sim)
            st.session_state.replay_index = len(sim["history"]) - 1
            st.session_state.replay_slider = st.session_state.replay_index
            st.rerun()
        if control_cols[3].button("Reset episode", width="stretch"):
            cfg = st.session_state.app_config
            st.session_state.simulation = new_simulation(
                strategy=cfg["strategy"],
                difficulty=cfg["difficulty"],
                visibility=cfg["visibility"],
                seed=cfg["seed"],
                start=cfg.get("start"),
                goal=cfg.get("goal"),
            )
            st.session_state.replay_index = 0
            st.session_state.pop("replay_slider", None)
            st.rerun()

        max_index = len(sim["history"]) - 1
        st.session_state.replay_index = st.select_slider(
            "Tactical replay · inspect an earlier observation",
            options=list(range(max_index + 1)),
            format_func=lambda index: f"Step {sim['history'][index]['step']} · {sim['history'][index]['event']}",
            key="replay_slider",
        )

    with intel_col:
        st.markdown('<div class="panel-heading"><h3>Intelligence report</h3><span class="panel-kicker">observation → action</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box"><b>Active policy:</b> {html.escape(sim["strategy"])}<br><b>Position:</b> {world.position}<br><b>Visible area:</b> {len(observation.visible_cells)} / {world.rows * world.cols} cells</div>', unsafe_allow_html=True)
        _score_panel(decision)
        st.markdown('<div class="panel-heading" style="margin-top:18px"><h3>Uncertainty radar</h3><span class="panel-kicker">unobserved cells by axis</span></div>', unsafe_allow_html=True)
        st.plotly_chart(uncertainty_radar(observation), width="stretch", config={"displayModeBar": False})
        st.caption("Directional counts show how many cells in each straight corridor remain unseen. They are counts, not threat probabilities.")

    st.markdown('<div class="panel-heading" style="margin-top:26px"><h3>Battle chronicle</h3><span class="panel-kicker">chronological · replayable</span></div>', unsafe_allow_html=True)
    recent = list(reversed(sim["history"][-8:]))
    rows = []
    for record in recent:
        if record["event"] == "start":
            text = "Mission briefing loaded; partial observation established."
        else:
            names = {
                "move": "Terrain traversed.",
                "wall": "A concealed barrier was struck and added to known terrain.",
                "hazard": "Hazard contact; location revealed.",
                "enemy": "Enemy interception; episode failed.",
                "goal": "Objective reached; mission complete.",
                "scan": "Information-gathering scan widened the revealed area.",
                "invalid": "Out-of-bounds or invalid action; no position change.",
                "terminal": "Episode already ended.",
            }
            text = names.get(record["event"], record["event"])
            if record["action"] != "—":
                text = f'{format_action(record["action"])} · {text}'
        rows.append(
            f'<div class="event-row"><span class="event-step">STEP {record["step"]:02d}</span>'
            f'<span class="event-text">{html.escape(text)}</span><span class="event-reward">{record["reward"]:+.1f}</span></div>'
        )
    st.markdown('<div class="event-list">' + "".join(rows) + "</div>", unsafe_allow_html=True)
    if world.status != "running":
        st.success(f"{world.termination_reason} Final reward: {world.cumulative_reward:.1f}.")


def render_strategy_lab() -> None:
    sim = st.session_state.simulation
    observation = current_observation(sim)
    st.markdown('<div class="eyebrow">Policies · decision audit</div><div class="page-title">Strategy Lab</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">All agents receive the same Observation object and share the same transition and reward rules. None can read hidden walls, traps, or enemies.</p>', unsafe_allow_html=True)
    cards = [
        ("Random Agent", "Uniform random baseline", "Selects randomly among moves not known to be blocked. A hidden wall can still create a collision."),
        ("Rule-Based", "Greedy safety heuristic", "Favors Manhattan progress, penalizes observed threats, and gives a small novelty reward to an unvisited cell."),
        ("Uncertainty-Aware AI", "Risk-weighted path planner", "Runs weighted Dijkstra using known walls, observed threats, and an explicit difficulty-based surcharge for unseen terrain."),
    ]
    cols = st.columns(3)
    for col, (name, label, description) in zip(cols, cards):
        with col:
            active = sim["strategy"] == name
            st.markdown(
                f'<div class="insight-box" style="min-height:174px;border-left-color:{"#e4c270" if active else "#6f5937"}">'
                f'<div class="eyebrow">{"ACTIVE POLICY" if active else "AVAILABLE POLICY"}</div>'
                f'<h3 style="margin:9px 0 5px;font:1.2rem Georgia;color:#ead4a0">{html.escape(name)}</h3>'
                f'<div style="color:#ba9b59;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase">{html.escape(label)}</div>'
                f'<p style="font-size:.82rem;color:#c4b99f;line-height:1.55;margin-top:10px">{html.escape(description)}</p></div>',
                unsafe_allow_html=True,
            )
    decision = pending_decision(sim)
    st.markdown("### Current decision audit")
    left, right = st.columns([1, 1])
    with left:
        _score_panel(decision)
    with right:
        st.markdown('<div class="parchment-note"><h4>What the policy can see</h4><ul>'
                    f'<li>Agent at {observation.position}; objective at {observation.goal}.</li>'
                    f'<li>{len(observation.visible_cells)} explored cells; {observation.rows * observation.cols - len(observation.visible_cells)} remain unobserved.</li>'
                    f'<li>{len(observation.visible_walls)} known walls; {len(observation.visible_hazards)} known hazards; {len(observation.visible_enemies)} known enemies.</li>'
                    '<li>Unknown cells receive a planning surcharge based on the selected difficulty, not a fabricated measured probability.</li>'
                    '</ul></div>', unsafe_allow_html=True)
        st.markdown("#### Legal alternatives")
        moves = legal_actions(observation)
        if moves:
            st.markdown(" · ".join(format_action(move) for move in moves))
        else:
            st.markdown("No adjacent move is known to be clear; scan or reset the episode.")
        st.caption("Legal here means within bounds and not a wall already seen. An unseen wall may still block a selected move.")

    st.markdown("---")
    st.markdown("### Common-seed tournament")
    st.caption("Every policy runs against the same set of seeds. Results are measured from actual episodes.")
    tournament_cols = st.columns([1, 1, 2])
    tournament_difficulty = tournament_cols[0].selectbox("Tournament difficulty", ("Easy", "Medium", "Hard"), index=("Easy", "Medium", "Hard").index(sim["world"].difficulty), key="tournament_difficulty")
    tournament_visibility = tournament_cols[1].selectbox("Visibility", ("High", "Partial", "Severe"), index=("High", "Partial", "Severe").index(sim["visibility"]), key="tournament_visibility")
    if tournament_cols[2].button("Run 8-seed tournament", width="stretch"):
        with st.spinner("Running common-seed episodes for each strategy…"):
            results = tournament(STRATEGIES, tournament_difficulty, tournament_visibility, sim["world"].seed, seeds=8)
        st.session_state.tournament_result = results
    result = st.session_state.get("tournament_result")
    if result is not None:
        st.dataframe(result, hide_index=True, width="stretch")
        st.caption("Leaderboard sorted by measured success rate, then average reward. Ties remain tied in the underlying metrics.")


def render_experiment_studio() -> None:
    st.markdown('<div class="eyebrow">Evaluation · measured outcomes</div><div class="page-title">Experiment Studio</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A reproducible comparison harness. Each selected strategy sees the same generated seeds at the selected visibility and difficulty.</p>', unsafe_allow_html=True)
    with st.form("experiment_configuration"):
        cols = st.columns([1, 1, 1, 1])
        episodes = cols[0].number_input("Seeds per strategy", min_value=1, max_value=300, value=12, step=1)
        difficulty = cols[1].selectbox("Difficulty", ("Easy", "Medium", "Hard"), index=1, key="exp_difficulty")
        visibility = cols[2].selectbox("Visibility", ("High", "Partial", "Severe"), index=1, key="exp_visibility")
        seed = cols[3].number_input("First seed", min_value=0, max_value=2_147_483_000, value=42, step=1)
        strategies = st.multiselect("Strategies to compare", STRATEGIES, default=list(STRATEGIES))
        submitted = st.form_submit_button("Run experiment", width="stretch")
    if submitted:
        if not strategies:
            st.error("Select at least one strategy.")
        else:
            with st.spinner(f"Running {int(episodes) * len(strategies)} seeded episodes…"):
                raw, summary, config = run_experiment(int(episodes), difficulty, visibility, strategies, int(seed))
            st.session_state.experiment_result = {"raw": raw, "summary": summary, "config": config}
    result = st.session_state.experiment_result
    if result is None:
        st.info("No experiment has been run in this session yet. Configure a comparison above to calculate measured outcomes.")
        return

    config = result["config"]
    st.markdown(
        f'<div class="insight-box"><b>Measured sample:</b> {config["total_episodes"]} episodes '
        f'({config["episodes_per_strategy"]} per strategy) · {html.escape(config["difficulty"])} · '
        f'{html.escape(config["visibility"])} visibility · seeds {config["seed_start"]}–{config["seed_start"] + config["episodes_per_strategy"] - 1}</div>',
        unsafe_allow_html=True,
    )
    summary = result["summary"]
    best = summary.sort_values(["success_rate", "average_reward"], ascending=False, kind="stable").iloc[0]
    ties = summary[(summary["success_rate"] == best["success_rate"]) & (summary["average_reward"] == best["average_reward"])]
    winners = ", ".join(ties["strategy"].astype(str).tolist())
    st.markdown(
        f'<div class="parchment-note" style="margin:14px 0"><h4>Observed result</h4>'
        f'<p>{html.escape(winners)} had the highest measured success rate in this sample '
        f'({best["success_rate"]:.1f}%). This describes these seeds only; it is not a guarantee of performance on other maps.</p></div>',
        unsafe_allow_html=True,
    )
    chart_cols = st.columns(3)
    chart_cols[0].plotly_chart(comparison_chart(summary, "success_rate", "Success rate", "%"), width="stretch", config={"displayModeBar": False})
    chart_cols[1].plotly_chart(comparison_chart(summary, "average_reward", "Average cumulative reward"), width="stretch", config={"displayModeBar": False})
    chart_cols[2].plotly_chart(comparison_chart(summary, "average_steps", "Average episode steps"), width="stretch", config={"displayModeBar": False})
    st.markdown("### Summary metrics")
    display = summary.rename(
        columns={
            "strategy": "Strategy",
            "sample_size": "Sample size",
            "success_rate": "Success rate (%)",
            "average_reward": "Avg reward",
            "average_steps": "Avg steps",
            "failure_rate": "Failure rate (%)",
            "collision_rate": "Collision rate (%)",
            "invalid_action_rate": "Invalid-action rate (%)",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")
    download_cols = st.columns(2)
    download_cols[0].download_button("Download episode results · CSV", csv_bytes(result["raw"]), "chakravyuha_episode_results.csv", "text/csv", width="stretch")
    download_cols[1].download_button("Download strategy summary · CSV", csv_bytes(summary), "chakravyuha_strategy_summary.csv", "text/csv", width="stretch")
    with st.expander("Inspect raw episode results"):
        st.dataframe(result["raw"], hide_index=True, width="stretch")
    st.caption("Collision rate = wall, hazard, and enemy contacts / action count. Invalid-action rate = rejected actions / action count. Rates are empirical, not model predictions.")


def render_architecture() -> None:
    st.markdown('<div class="eyebrow">System map · algorithm notes</div><div class="page-title">Architecture & Methods</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">The seven-stage pipeline below is implemented by these project modules; truth and agent observations remain separate.</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pipeline"><div class="pipeline-step"><strong>Environment</strong><code>environment.py</code><br><code>state.py</code></div>'
        '<div class="pipeline-step"><strong>Observation</strong><code>environment.py</code><br>visibility radius</div>'
        '<div class="pipeline-step"><strong>Decision</strong><code>agent.py</code><br><code>decision_engine.py</code></div>'
        '<div class="pipeline-step"><strong>Transition</strong><code>integration.py</code><br><code>environment.py</code></div>'
        '<div class="pipeline-step"><strong>Reward</strong><code>reward.py</code><br><code>metrics.py</code></div>'
        '<div class="pipeline-step"><strong>Evaluation</strong><code>evaluation.py</code><br><code>experiments.py</code></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Information boundaries")
    st.markdown(
        '<div class="insight-box"><b>True state:</b> walls, hazards, enemies, position, and objective are held by the environment. '
        '<b>Observation:</b> immutable view of explored cells, visible threats, current position, known objective, and scenario prior. '
        '<b>Decision:</b> receives the observation only. <b>Evaluator:</b> calculates outcomes from the environment after actions. '
        'The instructor display reads a separate replay truth snapshot and does not alter the agent input.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Algorithm and complexity")
    complexity = [
        ("Grid representation", "A 13 × 13 orthogonal lattice with a connected protected route and seeded walls/threats.", "O(R·C) generation"),
        ("Observation", "Manhattan-radius reveal; facts persist once explored.", "O(R·C) bounded scan"),
        ("Valid moves", "Four-neighbor bounds check and exclusion of known walls. Unknown terrain remains a possible move.", "O(1) per candidate"),
        ("Random Agent", "Seeded selection among candidate actions not known to be blocked.", "O(A)"),
        ("Rule-Based", "Manhattan progress, visible-threat penalties, and a small unexplored-cell preference.", "O(A)"),
        ("Uncertainty-Aware AI", "Weighted Dijkstra over the finite grid. Known walls are excluded; unseen cells receive a difficulty-based cost surcharge.", "O((V+E) log V) per decision"),
        ("Reward / metrics", "Deterministic transition reward and empirical episode aggregation.", "O(1) transition; O(N) aggregation"),
    ]
    st.dataframe(pd.DataFrame(complexity, columns=["Stage", "Implemented method", "Complexity"]), hide_index=True, width="stretch")
    st.caption("V = R·C cells, E ≤ 4V neighbor links, A ≤ 4 movement candidates. Search is finite and bounded by the grid; no neural network or reinforcement learning is claimed.")
    if st.session_state.experiment_result:
        st.markdown("### Latest measured comparison")
        summary = st.session_state.experiment_result["summary"]
        st.dataframe(summary, hide_index=True, width="stretch")
    else:
        st.info("Run an experiment to show empirical results here; no outcome is assumed in advance.")


def render_research_notes() -> None:
    st.markdown('<div class="eyebrow">College project · technical documentation</div><div class="page-title">Research Notes</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Concise notes for a faculty demonstration and viva.</p>', unsafe_allow_html=True)
    sections = [
        ("Problem statement", "An agent must navigate a hazardous, layered environment when it cannot observe every cell. The system compares complete environment truth with a policy’s partial observation, choice, and resulting outcome."),
        ("Objectives", "Generate reproducible battlefields; reveal terrain locally; compare random, rule-based, and uncertainty-aware policies; record decisions; evaluate actual episodes using common seeds."),
        ("System architecture", "The environment owns the hidden state and transitions. Observation generation exposes explored facts only. The decision engine scores candidate actions. Integration records replay snapshots. Evaluation and experiments use the same environment, agents, and reward function as the live simulator."),
        ("AI methods", "The baseline samples among not-known-blocked actions. The rule-based policy is a local greedy heuristic. The uncertainty-aware policy uses weighted Dijkstra on the finite grid, excludes known walls, adds extra traversal cost to unseen terrain based on difficulty, and penalizes observed threats."),
        ("Incomplete information", "The mission objective and grid bounds are known. The agent observes cells within a Manhattan visibility radius; explored facts persist. Hidden walls can cause a bump, hazards cause a reward penalty, and a hidden enemy ends the episode. A scan consumes a turn and widens future observation."),
        ("Reward function", "Goal +100; ordinary move −1; scan −0.5; concealed wall collision −4; hazard −7; invalid/out-of-bounds action −5; enemy interception −20. A fixed step limit yields timeout."),
        ("Experimental design", "For each strategy, episodes run on the same consecutive integer seeds, difficulty, and visibility. Agent random streams are deterministically derived from each seed. Results are computed from completed simulator transitions; no leaderboard outcome is hard-coded."),
        ("Metrics", "Success and failure rates are episode proportions. Average cumulative reward and steps are arithmetic means. Collision rate counts wall, hazard, and enemy contacts divided by actions. Invalid-action rate counts rejected actions divided by actions. The table includes actual sample size."),
        ("Limitations", "Enemies are static; the map uses a protected route to keep generated scenarios playable; uncertainty is represented by a heuristic prior surcharge rather than a learned probability model; experiments are local and do not establish statistical significance."),
        ("Future work", "Add moving adversaries, configurable reward weights, confidence intervals and paired statistical tests, larger scenario families, and a learned belief model, then validate each addition against the transparent baseline."),
    ]
    for title, body in sections:
        with st.expander(title, expanded=title in {"Problem statement", "AI methods"}):
            st.write(body)
    st.markdown("### Reward reference")
    st.dataframe(pd.DataFrame(reward_breakdown()), hide_index=True, width="stretch")
    result = st.session_state.experiment_result
    if result:
        st.markdown("### Measured results from this session")
        st.dataframe(result["summary"], hide_index=True, width="stretch")
        st.caption(f'{result["config"]["total_episodes"]} actual episodes · {result["config"]["difficulty"]} · {result["config"]["visibility"]}.')
    else:
        st.info("No experiment result is available yet. This section does not invent demonstration numbers.")
    st.markdown("### Demonstration sequence")
    st.markdown(
        "1. Enter the battlefield and note the starting observation.  \n"
        "2. Advance one move and explain the recommendation from its candidate scores.  \n"
        "3. Toggle the instructor truth layer to compare hidden state with the policy view.  \n"
        "4. Scan or continue through the formation; use replay to revisit decisions.  \n"
        "5. Run a common-seed experiment and export the measured rows as CSV."
    )


def main() -> None:
    initialize_state()
    _inject_styles(st.session_state.presentation, st.session_state.entered)
    if not st.session_state.entered:
        show_landing()
        return
    sidebar_controls()
    sim = st.session_state.simulation
    app_header(sim)
    if st.session_state.presentation and st.button("Exit presentation mode", type="secondary"):
        st.session_state.presentation = False
        st.rerun()
    page = render_navigation()
    if page == "Command Centre":
        render_command_centre()
    elif page == "Strategy Lab":
        render_strategy_lab()
    elif page == "Experiment Studio":
        render_experiment_studio()
    elif page == "Architecture":
        render_architecture()
    else:
        render_research_notes()


if __name__ == "__main__":
    main()
