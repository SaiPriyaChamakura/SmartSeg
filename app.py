import os
import pandas as pd
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from utils import compute_abc_xyz
from hatz_client import generate_recommendations
from transformers import pipeline
summarizer = pipeline("text-generation", model="distilgpt2")
import ollama
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 🔒 Prevent GPU meta-tensor issue


# 🔍 Helper: summarize dataframe for AI grounding
# ------------------------------------------------------------
def summarize_inventory_for_ai(df):
    top_segments = df['Segment'].value_counts().head(5).to_dict()
    top_value = df.nlargest(5, 'Annual_Value')[['SKU','Annual_Value','ABC','XYZ']].to_dict(orient='records')
    avg_cv = df['CV'].mean().round(2)
    total = len(df)
    high_prio = int((df['Priority_Class']=='High').sum())
    az_items = df[(df['ABC']=='A') & (df['XYZ']=='Z')].shape[0]

    summary = f"""
Inventory Summary:
- Total SKUs: {total}
- Average CV: {avg_cv}
- Top Segments (count): {top_segments}
- High-priority SKUs: {high_prio}
- A-Z risk items: {az_items}
Top 5 high-value SKUs:
{top_value}
"""
    return summary
# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title='SmartSeg — ABC × XYZ Classifier', layout='wide', page_icon='🧠')
st.title('SmartSeg ~ AI-Powered ABC × XYZ Inventory Classifier')
st.markdown('Upload your inventory CSV (columns: `SKU`, `Description`, `UnitCost`, `Jan..Dec`).')
st.markdown("We’ll compute **ABC (value)** and **XYZ (demand variability)** classes and optionally generate AI policy recommendations.")

# ------------------------------------------------------------
# SIDEBAR: THRESHOLDS & SETTINGS
# ------------------------------------------------------------
with st.sidebar:
    st.header('Controls')
    abc_A = st.slider('A threshold (cum %)', 60, 85, 70)
    abc_B = st.slider('B threshold (cum %)', 86, 98, 90)
    xyz_X = st.slider('X threshold (CV)', 0.1, 2.0, 0.5, 0.05)
    xyz_Y = st.slider('Y threshold (CV)', 0.2, 3.0, 1.0, 0.05)
    st.caption('Tip: Lower CV thresholds → stricter X class.')

# ------------------------------------------------------------
# FILE UPLOAD SECTION (Load button + sample fallback)
# ------------------------------------------------------------
uploaded = st.file_uploader('Upload CSV', type=['csv'])

if uploaded is None:
    st.info('👆 Upload a CSV file or try the bundled sample file below.')
    if st.button('📂 Load sample_data.csv'):
        st.session_state["raw_data"] = pd.read_csv('sample_data.csv')
        st.success("✅ Loaded bundled sample_data.csv successfully!")
else:
    if st.button('📥 Load Uploaded File'):
        st.session_state["raw_data"] = pd.read_csv(uploaded)
        st.success("✅ Uploaded dataset loaded successfully!")

if "raw_data" not in st.session_state:
    st.warning("Please upload or load a dataset to continue.")
    st.stop()

raw = st.session_state["raw_data"].copy()

# ------------------------------------------------------------
# VALIDATE MONTHLY COLUMNS
# ------------------------------------------------------------
month_prefixes = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_cols = [c for c in raw.columns if c[:3] in month_prefixes and c in raw.columns]
month_cols = [m for m in month_prefixes if m in month_cols]

if len(month_cols) != 12:
    st.error('❌ Expected 12 monthly demand columns (Jan..Dec). Please check your file.')
    st.stop()
else:
    st.success(f"✅ Data validated successfully — {len(month_cols)} monthly demand columns found.")

# ------------------------------------------------------------
# OPTIONAL FILTERS
# ------------------------------------------------------------
if "Category" in raw.columns:
    category_choice = st.selectbox("Filter by Category", ["All"] + sorted(raw["Category"].unique().tolist()))
    if category_choice != "All":
        raw = raw[raw["Category"] == category_choice]

if "Supplier" in raw.columns:
    supplier_choice = st.selectbox("Filter by Supplier", ["All"] + sorted(raw["Supplier"].unique().tolist()))
    if supplier_choice != "All":
        raw = raw[raw["Supplier"] == supplier_choice]

# Ensure Criticality exists
if "Criticality" not in raw.columns:
    np.random.seed(42)
    raw["Criticality"] = np.random.randint(1, 6, len(raw))

# ------------------------------------------------------------
# CLASSIFICATION (single call) + BASE TABLE
# ------------------------------------------------------------
result = compute_abc_xyz(
    raw,
    month_cols=month_cols,
    abc_thresholds={'A': abc_A, 'B': abc_B},
    xyz_thresholds={'X': xyz_X, 'Y': xyz_Y},
)

# Attach Criticality to result
result["Criticality_Score"] = raw["Criticality"].values

st.subheader('Segmentation Results')
st.dataframe(
    result[['SKU','Description','UnitCost','Annual_Usage','Annual_Value','CV','ABC','XYZ','Segment']],
    use_container_width=True
)

csv_bytes = result.to_csv(index=False).encode('utf-8')
st.download_button('⬇️ Download Classified CSV', data=csv_bytes,
                   file_name='classified_inventory.csv', mime='text/csv')

# ------------------------------------------------------------
# PARETO CHART: Annual Spend by SKU (SKU on X-axis)
# ------------------------------------------------------------
st.markdown("### Pareto Analysis of Annual Spend by SKU")

pareto_df = result.sort_values("Annual_Value", ascending=False).reset_index(drop=True)
pareto_df["CumValue"] = pareto_df["Annual_Value"].cumsum()
pareto_df["Cum%"] = pareto_df["CumValue"] / pareto_df["Annual_Value"].sum() * 100

fig, ax1 = plt.subplots(figsize=(10,4))
ax1.bar(pareto_df["SKU"], pareto_df["Annual_Value"], alpha=0.7)
ax1.set_xlabel("SKU (sorted by Annual Spend)")
ax1.set_ylabel("Annual Spend ($)")
ax2 = ax1.twinx()
ax2.plot(pareto_df["SKU"], pareto_df["Cum%"], color="red", marker="o")
ax2.axhline(70, color="green", linestyle="--", label="70% threshold")
ax2.set_ylabel("Cumulative % of Total Spend", color="red")
ax2.set_ylim(0,110)
ax2.legend(loc="lower right")
plt.xticks(rotation=60, ha="right")
st.pyplot(fig, use_container_width=True)

# ------------------------------------------------------------
# SAFETY STOCK & REORDER POINT
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 Reorder Policy Settings")
    enable_ss = st.checkbox("Enable Safety Stock & ROP", value=True)
    service_level = st.slider("Service Level (%)", 80, 99, 95)
    lead_time_days = st.number_input("Lead Time (days)", value=14)
    days_in_month = 30

    st.markdown("### ⚙️ Priority Weight Settings")
    w_abc = st.slider("Weight: ABC", 0.0, 1.0, 0.5, 0.05)
    w_xyz = st.slider("Weight: XYZ", 0.0, 1.0, 0.3, 0.05)
    w_crit = st.slider("Weight: Criticality", 0.0, 1.0, 0.2, 0.05)

if enable_ss:
    z_table = {90:1.28, 95:1.65, 97:1.88, 99:2.33}
    z = z_table.get(int(service_level), 1.65)

    avg_demand = raw[month_cols].mean(axis=1)
    std_demand = raw[month_cols].std(axis=1)

    lt_months = max(lead_time_days / days_in_month, 1e-9)
    result["Safety_Stock"] = (z * std_demand * np.sqrt(lt_months)).round(2)
    result["ROP"] = (avg_demand * lt_months + result["Safety_Stock"]).round(2)

    st.success(f"✅ Calculated Safety Stock & ROP using Z={z}, LT={lead_time_days} days")
    st.dataframe(result[["SKU","Segment","Annual_Usage","CV","Safety_Stock","ROP"]],
                 use_container_width=True)

# ------------------------------------------------------------
# HYBRID PRIORITY SCORING (ABC + XYZ + CRITICALITY)
# ------------------------------------------------------------
st.markdown("### Hybrid Priority Scoring (ABC + XYZ + Criticality)")

abc_map = {"A": 3, "B": 2, "C": 1}
xyz_map = {"X": 3, "Y": 2, "Z": 1}

result["ABC_Score"] = result["ABC"].map(abc_map)
result["XYZ_Score"] = result["XYZ"].map(xyz_map)

total_w = max(w_abc + w_xyz + w_crit, 1e-9)
result["Priority_Score"] = (
    w_abc * result["ABC_Score"]
    + w_xyz * result["XYZ_Score"]
    + w_crit * result["Criticality_Score"]
) / total_w

result["Priority_Class"] = pd.qcut(result["Priority_Score"], 3, labels=["Low","Medium","High"])

st.dataframe(
    result[["SKU","Segment","Criticality_Score","Priority_Score","Priority_Class"]],
    use_container_width=True
)

# Ensure urgency column exists for simulation
if "Urgency_Factor" not in result.columns:
    result["Urgency_Factor"] = 0

# ------------------------------------------------------------
# DYNAMIC SKU SIMULATION PANEL (dropdown controls)
# ------------------------------------------------------------
st.markdown("## ⚙️ Dynamic SKU Simulation Panel")

selected_sku = st.selectbox("Select an SKU to simulate", result["SKU"].tolist())
sku_row = result[result["SKU"] == selected_sku].iloc[0]

st.markdown(
    f"**Current Classification:** {sku_row['ABC']}-{sku_row['XYZ']}  |  "
    f"**Criticality:** {sku_row['Criticality_Score']}  |  "
    f"**Priority:** {sku_row['Priority_Class']}"
)

crit_levels = {
    "Low (1)": 1, "Medium (2)": 2, "High (3)": 3, "Very High (4)": 4, "Extreme (5)": 5
}
urg_levels = {
    "Normal (0)": 0, "Expedite (1)": 1, "Critical (2)": 2
}

crit_idx = max(0, min(int(sku_row["Criticality_Score"]) - 1, 4))
urg_idx = max(0, min(int(sku_row.get("Urgency_Factor", 0)), 2))

new_crit_label = st.selectbox("Select Criticality Level", list(crit_levels.keys()), index=crit_idx)
new_crit = crit_levels[new_crit_label]

new_urg_label = st.selectbox("Select Urgency Level", list(urg_levels.keys()), index=urg_idx)
new_urg = urg_levels[new_urg_label]

base_priority = (
    0.5 * abc_map[sku_row["ABC"]] +
    0.3 * xyz_map[sku_row["XYZ"]] +
    0.2 * new_crit
)
adjusted_priority = min(base_priority + (new_urg * 0.8), 5)

if adjusted_priority >= 4:
    new_class = "High"
elif adjusted_priority >= 2.5:
    new_class = "Medium"
else:
    new_class = "Low"

st.success(f"""
### 🔁 Updated Priority Simulation
**New Criticality:** {new_crit_label}  
**Urgency:** {new_urg_label}  
**Adjusted Priority Score:** {adjusted_priority:.2f}  
**Reclassified Priority Level:** {new_class}
""")

# ------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("A-Class %", f"{(result['ABC'].eq('A').mean()*100):.1f}%")
col2.metric("Z-Class %", f"{(result['XYZ'].eq('Z').mean()*100):.1f}%")
col3.metric("High-Priority Items", int((result['Priority_Class']=='High').sum()))

# ------------------------------------------------------------
# HEATMAP (auto-updating with metric selector) — uses computed model_df
# ------------------------------------------------------------
# Persist the fully computed dataframe so downstream sections share it
st.session_state["model_df"] = result.copy()

st.markdown("### 🗺️ ABC–XYZ Heatmap")
metric_option = st.selectbox(
    "Select metric to visualize:",
    ["Total Annual Spend ($)", "SKU Count", "Average Criticality", "Average Priority Score"],
    key="metric_selector"
)

model_df = st.session_state["model_df"]

# Make sure expected columns exist (defensive)
for col in ["Annual_Value", "Criticality_Score", "Priority_Score"]:
    if col not in model_df.columns:
        model_df[col] = 0

if metric_option == "Total Annual Spend ($)":
    values_col = "Annual_Value"; agg = "sum"; label = "Annual Spend ($)"
elif metric_option == "SKU Count":
    values_col = "SKU"; agg = "count"; label = "Count of SKUs"
elif metric_option == "Average Criticality":
    values_col = "Criticality_Score"; agg = "mean"; label = "Avg. Criticality"
else:
    values_col = "Priority_Score"; agg = "mean"; label = "Avg. Priority Score"

pivot = (
    model_df.pivot_table(
        index="ABC",
        columns="XYZ",
        values=values_col,
        aggfunc=agg,
        fill_value=0
    )
    .reindex(index=["A","B","C"], columns=["X","Y","Z"])
)

fig_hm = px.imshow(
    pivot,
    text_auto=True,
    color_continuous_scale="YlGnBu",
    aspect="auto",
    title=f"ABC–XYZ Matrix — {metric_option}"
)
fig_hm.update_traces(
    hovertemplate="ABC: %{y}<br>XYZ: %{x}<br>" + f"{label}: %{z:,.2f}<extra></extra>"
)
fig_hm.update_layout(
    xaxis_title="XYZ Class (Demand Variability)",
    yaxis_title="ABC Class (Annual Value)",
    yaxis=dict(autorange="reversed"),
    margin=dict(l=40, r=40, t=60, b=40),
    coloraxis_colorbar=dict(title=label)
)
st.plotly_chart(fig_hm, use_container_width=True)

# ------------------------------------------------------------

# ------------------------------------------------------------
# 🤖 Deterministic Insight Generator (No LLM, no extra deps)
# ------------------------------------------------------------
st.markdown("## Inventory Insights")

def generate_insights(df):
    out = []
    total = len(df)
    if total == 0:
        return ["No data loaded."]

    a_share = df['ABC'].eq('A').mean()*100
    z_share = df['XYZ'].eq('Z').mean()*100
    spend_top = (df.loc[df['ABC']=='A','Annual_Value'].sum() / df['Annual_Value'].sum()*100) if df['Annual_Value'].sum()>0 else 0
    high_priority = int((df['Priority_Class']=='High').sum())
    urgent = int((df.get('Urgency_Factor', 0) > 0).sum())

    # risk pockets
    az = df[(df['ABC']=='A') & (df['XYZ']=='Z')]
    bz = df[(df['ABC']=='B') & (df['XYZ']=='Z')]
    cx = df[(df['ABC']=='C') & (df['XYZ']=='X')]

    out.append(f"• A-class = {a_share:.1f}% of items but {spend_top:.1f}% of annual spend — focus review cadence here.")
    out.append(f"• Z-variability items = {z_share:.1f}% — stabilize demand/supply on these to cut expedites.")
    if len(az)>0:
        out.append(f"• High-risk pocket: A-Z items ({len(az)} SKUs). Raise safety stock temporarily and tighten supplier SLAs.")
    if len(bz)>0:
        out.append(f"• Watch B-Z items ({len(bz)} SKUs). Consider min-max with higher reorder multiplier during peaks.")
    if len(cx)>0:
        out.append(f"• Low-value but stable: C-X items ({len(cx)} SKUs). Lower service level or extend review interval to free cash.")
    out.append(f"• High-priority SKUs now: {high_priority}. Urgent overrides active on {urgent} SKUs.")
    return out

if st.button("✨ Generate Insights"):
    lines = generate_insights(st.session_state["model_df"])
    st.success("✅ Insights ready")
    for l in lines:
        st.write(l)


##################################################
# ------------------------------------------------------------
# 📦 Policy Template Engine (Deterministic)
# ------------------------------------------------------------
st.markdown("### 📦 Policy Recommendations")

POLICY_BY_SEGMENT = {
    "A-X": "Review 2–3x/week; Min–Max with tight bands; expedite triggers on consumption spike.",
    "A-Y": "Weekly review; EOQ or Min–Max; buffer +15–25% until variability improves.",
    "A-Z": "Daily review; Kanban or fixed lot; +30–50% safety stock; supplier collaboration on lead-time.",
    "B-X": "Weekly review; EOQ; gradual safety stock reduction.",
    "B-Y": "Biweekly review; Min–Max; seasonal multiplier as needed.",
    "B-Z": "Weekly review; dynamic safety multiplier; consider alternate suppliers.",
    "C-X": "Monthly review; wide Min–Max; lower service level target.",
    "C-Y": "Monthly; consolidate orders to reduce cost.",
    "C-Z": "Monthly/On-demand; consider make-to-order; avoid overstock."
}

def tailor_policy(base, crit, urg):
    tweaks = []
    if crit >= 4:
        tweaks.append("Increase service level target by 2–5 pts for critical use.")
    if urg == 1:
        tweaks.append("Temporarily pull next order forward; raise reorder multiple.")
    if urg == 2:
        tweaks.append("Activate emergency policy: daily review + temporary +25–40% safety stock.")
    if not tweaks:
        tweaks.append("Keep standard policy; monitor for 2 cycles.")
    return base + " " + " ".join(tweaks)

sku_for_policy = st.selectbox("Select SKU for policy", st.session_state["model_df"]["SKU"].tolist(), key="policy_sku")

if st.button("🧩 Generate Policy"):
    row = st.session_state["model_df"].loc[st.session_state["model_df"]["SKU"]==sku_for_policy].iloc[0]
    seg = f"{row['ABC']}-{row['XYZ']}"
    base = POLICY_BY_SEGMENT.get(seg, "Weekly review; Min–Max default policy.")
    crit = int(row.get("Criticality_Score", 3))
    urg = int(row.get("Urgency_Factor", 0))
    text = tailor_policy(base, crit, urg)
    st.info(f"**SKU {sku_for_policy} — {seg}**\n\n{text}")



# ------------------------------------------------------------
# 💬 Ask Your Data (Dynamic Q&A)
# ------------------------------------------------------------
st.markdown("### 💬 Ask Your Data (AI-powered)")

user_query = st.text_area("Ask a question about your inventory:",
                          placeholder="Example: Which SKUs are driving most variability or cost?")

if st.button("🧩 Ask AI with Context"):
    context = summarize_inventory_for_ai(st.session_state["model_df"])
    question_prompt = f"""
You are an operations analyst. Here is the current inventory data summary:
{context}

Answer the following user question concisely, referring to real values or SKUs where relevant:
{user_query}
"""
    try:
        answer = ollama.chat(model="phi", messages=[
            {"role": "system", "content": "You are a factual and structured AI assistant for inventory analytics."},
            {"role": "user", "content": question_prompt}
        ])
        st.info(answer["message"]["content"])
    except Exception as e:
        st.error(f"Ollama Q&A failed: {e}")



