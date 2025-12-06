# app.py
import streamlit as st
import pandas as pd
import math
from functools import lru_cache
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from pulp import (
    LpProblem, LpVariable, LpMinimize, lpSum, value,
    LpStatus, PULP_CBC_CMD, GLPK_CMD, listSolvers
)
from pulp import COIN_CMD


st.set_page_config(page_title="Shelter Optimization Dashboard", layout="wide")

st.title("Shelter Optimization — Interactive Dashboard")
st.markdown(
    "Adjust model parameters on the left. The model runs in the background and shows decisions, KPIs, and sensitivity."
)

#Definition of ZIP Codes in the county
ORIGINAL_ZIPS = [33012, 33157, 33186, 33015, 33033, 33178, 33142, 33032, 33177, 33018, 33125, 33196, 33161, 33176, 33165, 33175, 33162, 33179, 33193, 33147, 33126, 33016, 33155, 33169, 33010, 33160, 33014, 33172, 33055, 33134, 33056, 33030, 33141, 33139, 33135, 33133, 33174, 33183, 33173, 33130, 33180, 33156, 33150, 33143, 33054, 33013, 33185, 33145, 33138, 33127, 33144, 33166, 33137, 33189, 33034, 33168, 33167, 33131, 33184, 33181, 33140, 33187, 33146, 33132, 33190, 33136, 33035, 33129, 33149, 33154, 33170, 33182, 33128, 33194, 33031, 33158, 33122, 33109, 33101, 33039, ]
ZIPS = ORIGINAL_ZIPS.copy()
I = J = ZIPS

manual_populations = {
    33012: 71088, 33157: 69152, 33186: 68750, 33015: 68517, 33033: 68121,
    33178: 64751, 33142: 59121, 33032: 57327, 33177: 56791, 33018: 55241,
    33125: 54873, 33196: 54187, 33161: 52824, 33176: 52719, 33165: 52047,
    33175: 50233, 33162: 48326, 33179: 48196, 33193: 47116, 33147: 47065,
    33126: 46798, 33016: 45620, 33155: 43702, 33169: 42156, 33010: 42081,
    33160: 42058, 33014: 40739, 33172: 40278, 33055: 37977, 33134: 37958,
    33056: 37274, 33030: 36405, 33141: 35967, 33139: 35100, 33135: 34742,
    33133: 34500, 33174: 34335, 33183: 34057, 33173: 33026, 33130: 32946,
    33180: 32881, 33156: 32582, 33150: 31807, 33143: 31688, 33054: 31590,
    33013: 29830, 33185: 29530, 33145: 29240, 33138: 28609, 33127: 28306,
    33144: 26596, 33166: 26364, 33137: 25283, 33189: 24575, 33034: 23774,
    33168: 23458, 33167: 22823, 33131: 22540, 33184: 20850, 33181: 20448,
    33140: 19785, 33187: 19346, 33146: 17962, 33132: 17136, 33190: 16909,
    33136: 16854, 33035: 16788, 33129: 15184, 33149: 14639, 33154: 14557,
    33170: 14477, 33182: 13163, 33128: 8989, 33194: 8617, 33031: 7692,
    33158: 6536, 33122: 1873, 33109: 792
}

# user input
st.sidebar.header("Model Parameters")
alpha_pct = st.sidebar.slider("Minimum coverage per ZIP (%)", min_value=5, max_value=25, value=10, step=1)
alpha = alpha_pct / 100.0

# capacity default computed below if user doesn't input
C_input = st.sidebar.number_input("Shelter capacity (C) — leave 0 to use default", min_value=0, value=0, step=1)

Dmax = st.sidebar.slider("Maximum travel distance (miles)", min_value=1.0, max_value=50.0, value=12.5, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("Solver options")
solver_choice = st.sidebar.selectbox(
    "Solver",
    options=[
        "PuLP default (m.solve())",
        "Auto (PULP_CBC_CMD)",
        "CBC: provide path",
        "GLPK_CMD",
    ],
)
cbc_path = ""
if solver_choice == "CBC: provide path":
    cbc_path = st.sidebar.text_input(
        "Path to cbc.exe (e.g. C:\\cbc\\cbc.exe)", value=""
    )


st.sidebar.markdown("---")
st.sidebar.header("Advanced")
n_exp = st.sidebar.number_input("Expected number of shelters (n_exp)", min_value=1, max_value=200, value=30, step=1)
run_btn = st.sidebar.button("Run model")

# file uploader for population or coords if user wants
st.sidebar.markdown("**Optional uploads**")
pop_file = st.sidebar.file_uploader("Upload population CSV (columns: ZIP,pop)", type=["csv"])
coords_file = st.sidebar.file_uploader("Upload coords CSV (columns: ZIP,lat,lon)", type=["csv"])

# censusdata information
@st.cache_data(show_spinner=False)
def prepare_populations(zips, uploaded_pop_file):
    pop = {}
    # Try importing censusdata, but fall back to manual_populations if unavailable
    try:
        import censusdata
        # attempt to fetch for zips (may be rate limited)
        for z in zips:
            pop[z] = 0
        # We'll attempt to download, but if it fails we'll populate later
        for z in zips:
            try:
                geo = censusdata.censusgeo([('zip code tabulation area', str(z))])
                data = censusdata.download('acs5', 2022, geo, ['B01003_001E'])
                pop[z] = int(data['B01003_001E'].values[0])
            except Exception:
                pop[z] = 0
    except Exception:
        # censusdata not available locally; use manual map or default
        for z in zips:
            pop[z] = manual_populations.get(z, 33480)

    # If the user uploaded a population CSV, override
    if uploaded_pop_file is not None:
        try:
            dfp = pd.read_csv(uploaded_pop_file)
            for _, row in dfp.iterrows():
                z = int(row['ZIP'])
                pop[z] = int(row['pop'])
        except Exception:
            st.warning("Uploaded pop CSV could not be parsed. Expected columns: ZIP,pop")

    # Fill zeros using manual_populations or average
    avg_pop = int(sum(pop.values()) / len(zips))
    for z in zips:
        if pop.get(z, 0) == 0:
            pop[z] = manual_populations.get(z, avg_pop)
    return pop

pop = prepare_populations(ZIPS, pop_file)

# geocoder information
@st.cache_data(show_spinner=False)
def geocode_zips(zips, uploaded_coords_file=None):
    coords = {}
    # If coordinates uploaded, use those
    if uploaded_coords_file is not None:
        try:
            dfc = pd.read_csv(uploaded_coords_file)
            for _, row in dfc.iterrows():
                z = int(row['ZIP'])
                coords[z] = (float(row['lat']), float(row['lon']))
            # Fill missing zips with None
            for z in zips:
                if z not in coords:
                    coords[z] = None
            return coords, True
        except Exception:
            st.warning("Uploaded coords CSV could not be parsed. Expected columns: ZIP,lat,lon")

    geolocator = Nominatim(user_agent="shelter_dashboard_geocoder")
    for z in zips:
        try:
            loc = geolocator.geocode(f"{z} Miami-Dade County FL USA", timeout=10)
            if loc:
                coords[z] = (loc.latitude, loc.longitude)
            else:
                coords[z] = None
        except (GeocoderTimedOut, GeocoderUnavailable):
            coords[z] = None
        except Exception:
            coords[z] = None
    # return coords and a boolean saying whether geocoding succeeded for all
    all_ok = all(coords[z] is not None for z in zips)
    return coords, all_ok

coords, all_geocoded = geocode_zips(ZIPS, coords_file)

if not all_geocoded:
    st.warning(
        "Note: Some ZIP geocodes failed. You can upload a coords CSV (columns: ZIP,lat,lon) to avoid geocoding rate limits."
    )

# distances and accessibility
@st.cache_data(show_spinner=False)
def build_distance_and_a(zips, coords, Dmax):
    dist = {}
    a = {}
    for i in zips:
        for j in zips:
            if coords.get(i) is None or coords.get(j) is None:
                dist[(i, j)] = float('inf')
                a[(i, j)] = 0
            else:
                d = geodesic(coords[i], coords[j]).miles
                dist[(i, j)] = d
                a[(i, j)] = 1 if d <= Dmax else 0
    return dist, a

dist, a_default = build_distance_and_a(ZIPS, coords, Dmax)

# Compute defaults for C if user left 0
Ctotal = round(0.125 * sum(pop[z] for z in ZIPS))
if C_input <= 0:
    C = max(1, round(Ctotal / n_exp))
else:
    C = C_input

# List of existing shelters (E) and e dict (as in your code)
E = [
    33179, 33180, 33162, 33161, 33147, 33056, 33015, 33018, 33136,
    33142, 33178, 33165, 33175, 33155, 33173, 33196, 33177, 33030
]
e = {z: (1 if z in E else 0) for z in ZIPS}

# function to solve model
def build_and_solve_model(alpha_val, C_val, Dmax_val, solver_choice="Auto", cbc_path_str=""):
    # recompute a based on new Dmax
    a_local = { (i,j): (1 if dist[(i,j)] <= Dmax_val else 0) for i in ZIPS for j in ZIPS }

    # Define variables
    y = LpVariable.dicts("y", ZIPS, cat="Binary")
    x = LpVariable.dicts("x", [(i, j) for i in ZIPS for j in ZIPS], lowBound=0, cat="Continuous")
    zvar = LpVariable.dicts("z", [(i, j) for i in ZIPS for j in ZIPS], cat="Binary")

    m = LpProblem("Shelter_Optimization", LpMinimize)
    # objective
    m += lpSum(y[j] + e[j] for j in ZIPS)

    # constraints (translated from your Colab)
    for i in ZIPS:
        for j in ZIPS:
            m += x[(i, j)] <= a_local[(i, j)] * pop[i]
            m += x[(i, j)] <= pop[i] * (y[j] + e[j])
            m += zvar[(i, j)] <= a_local[(i, j)]
            m += zvar[(i, j)] <= y[j] + e[j]

    for j in ZIPS:
        m += y[j] + e[j] <= 1
        m += lpSum(x[(i, j)] for i in ZIPS) <= C_val * (y[j] + e[j])

    for i in ZIPS:
        m += lpSum(x[(i, j)] for j in ZIPS) >= alpha_val * pop[i]
        m += lpSum(x[(i, j)] for j in ZIPS) <= pop[i]
        m += lpSum(zvar[(i, j)] for j in ZIPS) == 1

    # Choose solver
    solver = None
    solver_msg = ""
    try:
        if solver_choice == "Auto (PULP_CBC_CMD)":
            # try default CBC (may fail on Windows if binary mismatched)
            solver = PULP_CBC_CMD(msg=0)
            solver_msg = "PULP_CBC_CMD (auto)"
        elif solver_choice == "CBC: provide path":
            if cbc_path_str:
                solver = COIN_CMD(path=cbc_path_str, msg=0)
                solver_msg = f"COIN_CMD (path={cbc_path_str})"
            else:
                raise ValueError("CBC path not provided.")
        elif solver_choice == "GLPK_CMD":
            solver = GLPK_CMD(msg=0)
            solver_msg = "GLPK_CMD"
    except Exception as ex:
        st.error(f"Error creating solver: {ex}")
        return {"status": -1, "error": f"Solver creation failed: {ex}"}

    # Solve and catch solver issues
    try:
        m.solve(solver)
    except Exception as ex:
        # Return error info and suggest pointing to local solver
        return {"status": -1, "error": f"Solver execution error: {ex}. Try providing a CBC path or installing GLPK."}

    # Check status
    stat = LpStatus[m.status]
    if m.status != 1:
        return {"status": m.status, "status_text": stat, "error": f"Solver finished with status {stat}"}

    # Extract solution
    y_open = {j: int(value(y[j])) for j in ZIPS}
    # include existing shelters (e[j] == 1) explicitly
    for j in ZIPS:
        if e[j] == 1:
            y_open[j] = 1

    # assignments: prefer zvar if available else x
    assignments = {}
    for i in ZIPS:
        assigned_j = None
        # try zvar
        for j in ZIPS:
            try:
                if zvar[(i, j)].value() is not None and int(round(zvar[(i, j)].value())) == 1:
                    assigned_j = j
                    break
            except Exception:
                pass
        if assigned_j is None:
            # fallback: pick j with max x
            max_x = -1
            for j in ZIPS:
                val = x[(i, j)].value()
                if val is None: 
                    val = 0
                if val > max_x:
                    max_x = val
                    assigned_j = j
        assignments[i] = assigned_j

    # compute utilization per shelter
    utilization = {}
    for j in ZIPS:
        total_assigned = sum((x[(i, j)].value() or 0) for i in ZIPS)
        utilization[j] = total_assigned

    objective_value = value(m.objective)
    return {
        "status": m.status,
        "status_text": stat,
        "solver": solver_msg,
        "objective": objective_value,
        "y_open": y_open,
        "assignments": assignments,
        "utilization": utilization,
        "model": m
    }

# ---------------------------
# Run model on button press or initial load
# ---------------------------
if run_btn:
    with st.spinner("Solving optimization..."):
        result = build_and_solve_model(alpha, C, Dmax, solver_choice=solver_choice, cbc_path_str=cbc_path)
    if result.get("status", -1) != 1:
        st.error(f"Model did not solve successfully: {result.get('error', 'Unknown error')} (status {result.get('status')})")
    else:
        st.success("Model solved.")
        # KPIs
        st.subheader("Key Performance Indicators")
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum shelters (objective)", result["objective"])
        col2.metric("Total population covered (approx)", sum(pop.values()))
        # count opened shelters (include existing e)
        opened = [j for j, val in result["y_open"].items() if val == 1]
        col3.metric("Shelters open (count)", len(opened))

        # Opened shelters table
        st.subheader("Opened Shelters")
        df_open = pd.DataFrame({
            "ZIP": opened,
            "ExistingShelter": [1 if z in E else 0 for z in opened],
            "Capacity": [C]*len(opened),
            "Utilization (assigned evacuees)": [int(result["utilization"].get(z, 0)) for z in opened]
        })
        st.table(df_open.sort_values("ZIP"))

        # Assignments table
        st.subheader("Assignments (which shelter each ZIP is assigned to)")
        df_assign = pd.DataFrame({
            "ZIP": list(result["assignments"].keys()),
            "Assigned_Shelter": list(result["assignments"].values()),
            "Population": [pop[z] for z in result["assignments"].keys()]
        })
        st.dataframe(df_assign.sort_values("ZIP"))

        # Sensitivity analysis quick visualization (alpha sweep)
        st.subheader("Quick sensitivity: alpha sweep (using current C & Dmax)")
        alpha_vals = [0.05, 0.10, 0.15, 0.20, 0.25]
        rows = []
        for a_val in alpha_vals:
            out = build_and_solve_model(a_val, C, Dmax, solver_choice=solver_choice, cbc_path_str=cbc_path)
            rows.append({"alpha": a_val, "objective": out.get("objective") if out.get("status")==1 else None})
        df_sens = pd.DataFrame(rows)
        st.line_chart(df_sens.rename(columns={"alpha":"index"}).set_index("alpha")["objective"])

else:
    st.info("Adjust parameters and click **Run model** in the sidebar to solve the optimization.")

# ---------------------------
# Footer: help + requirements
# ---------------------------
st.markdown("---")
st.markdown(
    "Troubleshooting\n\n"
    "- Windows --> use `cbc.exe` if PuLP is incompatible with your run of the model. "
    "Download CBC and enter in the Solver options. \n"
    "- If this still does not work, install GLPK and select `GLPK_CMD` in the Solver options.\n"
)



