import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import csv
import re

# Page configuration
st.set_page_config(page_title="LCD Gamma Simulator", layout="wide")

# ==========================================
# 1. Base Register Map Definition
# ==========================================
REGISTER_MAP_DEF = {
    "0XC7": {"name": "Gamma DAC (Analog Nodes)", "params": []},
    "0XC8": {"name": "VGMPHO / VGS_S", "params": []},
    "0XC9": {"name": "VGMNHO", "params": []},
    "0XCF": {"name": "Digital Gamma Nodes", "params": []},
}

# ==========================================
# 2. Safe CSV Loading and Parameter Parsing
# ==========================================
def safe_read_csv(file_obj):
    file_obj.seek(0)
    raw_bytes = file_obj.read()
    
    text = None
    for enc in ['utf-8', 'shift_jis', 'cp932', 'utf-16', 'latin-1']:
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
            
    if text is None:
        text = raw_bytes.decode('utf-8', errors='replace')
        
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    return pd.DataFrame(rows)

def parse_uploaded_register_defs(uploaded_files):
    new_defs = {}
    for file_obj in uploaded_files:
        try:
            df = safe_read_csv(file_obj)
            
            cmd_name_col = -1
            cmd_param_col = -1
            hex_col = -1
            d_cols = []

            for i, row in df.iterrows():
                row_upper_clean = [str(x).strip().upper().replace('\n', '') for x in row]
                
                if cmd_name_col == -1 and 'COMMAND NAME' in row_upper_clean:
                    cmd_name_col = row_upper_clean.index('COMMAND NAME')
                elif cmd_name_col == -1 and 'REGISTER NAME' in row_upper_clean:
                    cmd_name_col = row_upper_clean.index('REGISTER NAME')
                    
                if cmd_param_col == -1 and 'COMMAND/PARAMETER' in row_upper_clean:
                    cmd_param_col = row_upper_clean.index('COMMAND/PARAMETER')
                    
                if hex_col == -1 and 'HEX' in row_upper_clean:
                    hex_col = row_upper_clean.index('HEX')
                elif hex_col == -1 and 'ADDRESS' in row_upper_clean:
                    hex_col = row_upper_clean.index('ADDRESS')
                    
                if not d_cols and 'D7' in row_upper_clean and 'D0' in row_upper_clean:
                    for d in ['D7', 'D6', 'D5', 'D4', 'D3', 'D2', 'D1', 'D0']:
                        if d in row_upper_clean:
                            d_cols.append(row_upper_clean.index(d))
                            
                if cmd_name_col != -1 and cmd_param_col != -1 and hex_col != -1 and d_cols:
                    break
                    
            if cmd_name_col != -1 and cmd_param_col != -1 and hex_col != -1:
                current_cmd = None
                for i in range(len(df)):
                    row = df.iloc[i]
                    if len(row) <= max(cmd_param_col, hex_col):
                        continue
                        
                    cmd_param_val = str(row[cmd_param_col]).strip().lower().replace('\n', '')
                    hex_val = str(row[hex_col]).strip().upper()
                    
                    if 'command' in cmd_param_val and cmd_param_val != 'command/parameter':
                        if hex_val != '' and hex_val != 'XX' and hex_val != 'NAN' and len(hex_val) <= 4:
                            if not hex_val.startswith('0X'):
                                hex_val = '0X' + hex_val
                            current_cmd = hex_val
                            cmd_name = str(row[cmd_name_col]).strip() if len(row) > cmd_name_col else "Unknown"
                            if cmd_name.lower() == 'nan': cmd_name = "Unknown"
                            if current_cmd not in new_defs:
                                new_defs[current_cmd] = {"name": cmd_name, "params": []}
                                
                    elif 'parameter' in cmd_param_val and current_cmd:
                        param_names = []
                        for c in d_cols:
                            if c < len(row):
                                val = str(row[c]).strip()
                                if val and val.lower() not in ['0', '1', '0.0', '1.0', '-', 'nan', 'xx']:
                                    clean_val = re.sub(r'\[.*?\]', '', val).strip()
                                    if clean_val and clean_val not in param_names:
                                        param_names.append(clean_val)
                                        
                        param_name_str = " / ".join(param_names) if param_names else f"Param_{len(new_defs[current_cmd]['params']) + 1}"
                        new_defs[current_cmd]["params"].append(param_name_str)
                        
        except Exception as e:
            st.sidebar.error(f"Failed to load definition from {file_obj.name}: {e}")
            
    return new_defs

# ==========================================
# TD7875 Hardware Constants & Formulas
# ==========================================
X_POINTS = np.array([0, 4, 15, 27, 43, 67, 91, 111, 119, 128, 152, 176, 192, 212, 228, 243, 250, 254, 255], dtype=float)
OFFSETS = np.array([0, 0, 0, 0, 256, 256, 256, 512, 512, 512, 512, 769, 769, 769, 1025, 1025, 1025, 1025, 1025], dtype=float)

def apply_td7875_hardware_formulas(cp_dict):
    V = np.full(256, np.nan)
    for cp, val in cp_dict.items():
        V[int(cp)] = val

    def calc(n, top, bot, w_num, w_den):
        V[n] = (V[top] - V[bot]) * (w_num / w_den) + V[bot]

    calc(1, 0, 4, 7.6, 12.15); calc(2, 0, 4, 4.6, 12.15); calc(3, 0, 4, 2.0, 12.15)
    calc(5, 4, 15, 11.2, 13.15); calc(6, 4, 15, 9.7, 13.15); calc(7, 4, 15, 8.2, 13.15)
    calc(8, 4, 15, 7.0, 13.15); calc(9, 4, 15, 5.9, 13.15); calc(10, 4, 15, 4.8, 13.15)
    calc(11, 4, 15, 3.7, 13.15); calc(12, 4, 15, 2.7, 13.15); calc(13, 4, 15, 1.7, 13.15)
    calc(14, 4, 15, 0.8, 13.15)
    calc(19, 15, 27, 4.4, 7.2); calc(23, 15, 27, 2.1, 7.2)
    calc(31, 27, 43, 4.4, 6.05); calc(35, 27, 43, 2.8, 6.05); calc(39, 27, 43, 1.3, 6.05)
    calc(47, 43, 67, 7.1, 8.75); calc(51, 43, 67, 5.5, 8.75); calc(55, 43, 67, 4.1, 8.75)
    calc(59, 43, 67, 2.7, 8.75); calc(63, 43, 67, 1.3, 8.75)
    calc(71, 67, 91, 5.1, 6.25); calc(75, 67, 91, 4.05, 6.25); calc(79, 67, 91, 3.0, 6.25)
    calc(83, 67, 91, 1.9, 6.25); calc(87, 67, 91, 0.9, 6.25)
    calc(95, 91, 111, 3.9, 5.0); calc(99, 91, 111, 2.9, 5.0); calc(103, 91, 111, 1.9, 5.0)
    calc(107, 91, 111, 0.9, 5.0)
    calc(115, 111, 119, 1.0, 2.0)
    calc(123, 119, 128, 1.25, 2.25); calc(127, 119, 128, 0.25, 2.25)
    calc(132, 128, 152, 5.0, 6.0); calc(136, 128, 152, 4.0, 6.0); calc(140, 128, 152, 3.0, 6.0)
    calc(144, 128, 152, 2.0, 6.0); calc(148, 128, 152, 1.0, 6.0)
    calc(156, 152, 176, 5.05, 6.0); calc(160, 152, 176, 4.1, 6.0); calc(164, 152, 176, 3.1, 6.0)
    calc(168, 152, 176, 2.1, 6.0); calc(172, 152, 176, 1.075, 6.0)
    calc(180, 176, 192, 3.05, 4.0); calc(184, 176, 192, 2.075, 4.0); calc(188, 176, 192, 1.05, 4.0)
    calc(196, 192, 212, 4.95, 6.05); calc(200, 192, 212, 3.8, 6.05); calc(204, 192, 212, 2.6, 6.05)
    calc(208, 192, 212, 1.35, 6.05)
    calc(216, 212, 228, 4.3, 5.45); calc(220, 212, 228, 3.0, 5.45); calc(224, 212, 228, 1.575, 5.45)
    calc(232, 228, 243, 5.1, 6.5); calc(233, 228, 243, 4.7, 6.5); calc(234, 228, 243, 4.3, 6.5)
    calc(235, 228, 243, 3.9, 6.5); calc(236, 228, 243, 3.45, 6.5); calc(237, 228, 243, 3.0, 6.5)
    calc(238, 228, 243, 2.55, 6.5); calc(239, 228, 243, 2.1, 6.5); calc(240, 228, 243, 1.6, 6.5)
    calc(241, 228, 243, 1.1, 6.5); calc(242, 228, 243, 0.6, 6.5)
    calc(244, 243, 250, 7.1, 8.05); calc(245, 243, 250, 6.1, 8.05); calc(246, 243, 250, 5.0, 8.05)
    calc(247, 243, 250, 3.9, 8.05); calc(248, 243, 250, 2.7, 8.05); calc(249, 243, 250, 1.3, 8.05)
    calc(251, 250, 254, 10.7, 13.1); calc(252, 250, 254, 6.7, 13.1); calc(253, 250, 254, 3.0, 13.1)

    x_all = np.arange(256)
    valid_mask = ~np.isnan(V)
    V_full = np.interp(x_all, x_all[valid_mask], V[valid_mask])
    return V_full

def convert_dac_to_physical_voltage(dac_values, polarity, v_gmp, v_gmn, v_gss):
    Y = dac_values + OFFSETS
    ratio = 1.0 - (Y / 2048.0)
    if polarity == 'positive':
        return v_gss + (v_gmp - v_gss) * ratio
    else:
        return v_gss - (v_gss - v_gmn) * ratio

def convert_voltage_to_dac_pos(v_target_array, v_gmp, v_gss):
    ratio = (v_target_array - v_gss) / (v_gmp - v_gss)
    Y = 2048.0 * (1.0 - ratio)
    dac_values = np.round(Y - OFFSETS).clip(0, 1023).astype(int)
    return dac_values

def process_td7875_physical_tuning(meas_gray, meas_lum, init_dac_array, v_gmp, v_gmn, v_gss, target_gamma=2.2):
    max_lum = np.max(meas_lum)
    x_cont = np.linspace(0, 255, 256)

    v_pos_init_cp = convert_dac_to_physical_voltage(init_dac_array, 'positive', v_gmp, v_gmn, v_gss)
    v_pos_full_init = apply_td7875_hardware_formulas(dict(zip(X_POINTS, v_pos_init_cp)))
    
    v_neg_init_cp = convert_dac_to_physical_voltage(init_dac_array, 'negative', v_gmp, v_gmn, v_gss)
    v_neg_full_init = apply_td7875_hardware_formulas(dict(zip(X_POINTS, v_neg_init_cp)))

    meas_v_applied = np.interp(meas_gray, x_cont, v_pos_full_init)
    
    target_lum_cp = ((X_POINTS / 255.0) ** target_gamma) * max_lum
    target_lum_cont = ((x_cont / 255.0) ** target_gamma) * max_lum

    sort_lum_idx = np.argsort(meas_lum)
    target_v_cp = np.interp(target_lum_cp, meas_lum[sort_lum_idx], meas_v_applied[sort_lum_idx])

    adj_dac_array = convert_voltage_to_dac_pos(target_v_cp, v_gmp, v_gss)

    v_pos_adj_cp = convert_dac_to_physical_voltage(adj_dac_array, 'positive', v_gmp, v_gmn, v_gss)
    v_pos_full_adj = apply_td7875_hardware_formulas(dict(zip(X_POINTS, v_pos_adj_cp)))

    v_neg_adj_cp = convert_dac_to_physical_voltage(adj_dac_array, 'negative', v_gmp, v_gmn, v_gss)
    v_neg_full_adj = apply_td7875_hardware_formulas(dict(zip(X_POINTS, v_neg_adj_cp)))

    sort_v_idx = np.argsort(meas_v_applied)
    lum_adjusted = np.interp(v_pos_full_adj, meas_v_applied[sort_v_idx], meas_lum[sort_v_idx])
    lum_meas_continuous = np.interp(v_pos_full_init, meas_v_applied[sort_v_idx], meas_lum[sort_v_idx])

    def calc_gamma(lum_array):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(lum_array / max_lum) / np.log(x_cont / 255.0)

    return {
        "max_lum": max_lum, "x_cont": x_cont, "cp_x": X_POINTS,
        "init_dac": init_dac_array, "adj_dac": adj_dac_array,
        "cp_names": [f"V{int(x)}P/N" for x in X_POINTS],
        "v_pos_full_init": v_pos_full_init, "v_pos_full_adj": v_pos_full_adj,
        "v_neg_full_init": v_neg_full_init, "v_neg_full_adj": v_neg_full_adj,
        "v_pos_init_cp": v_pos_init_cp, "v_pos_adj_cp": v_pos_adj_cp,
        "v_neg_init_cp": v_neg_init_cp, "v_neg_adj_cp": v_neg_adj_cp,
        "lum_meas": lum_meas_continuous, "lum_adj": lum_adjusted,
        "lum_tgt": target_lum_cont,
        "gam_meas": calc_gamma(lum_meas_continuous), 
        "gam_adj": calc_gamma(lum_adjusted), 
        "gam_tgt": np.full_like(x_cont, target_gamma)
    }

# ==========================================
# Streamlit UI Construction
# ==========================================
st.title("LCD Gamma Simulator")
st.markdown("Strictly decodes the **Register Map (0xC7-0xCF)** to simulate physical voltages and digital tuning parameters.")

# Sidebar: Settings
with st.sidebar:
    # --- ロゴの追加 ---
    try:
        st.image("yitoa.png", width="stretch")
        st.markdown(
            "<div style='text-align: center; font-size: 13px; color: #6c757d; margin-bottom: 20px;'>"
            "Copyright(c) YITOA Technology.<br>All rights reserved."
            "</div>", 
            unsafe_allow_html=True
        )
    except Exception:
        pass 
        
    st.header("1. Basic Settings")
    target_gamma_input = st.number_input("Target Gamma", min_value=1.0, max_value=3.0, value=2.2, step=0.1)
    
    st.markdown("---")
    st.subheader("Optional: Upload Register Definition Maps")
    st.markdown("Dynamic extraction and mapping of register names from uploaded MCS/DCS CSV files.")
    def_files = st.file_uploader("Select Register Map Def (MCS/DCS)", type=["csv"], accept_multiple_files=True, key="def_uploader")
    if def_files:
        dynamic_defs = parse_uploaded_register_defs(def_files)
        REGISTER_MAP_DEF.update(dynamic_defs)
        if dynamic_defs:
            st.success(f"Successfully mapped {len(dynamic_defs)} command definitions!")

    st.markdown("---")
    st.subheader("2. Upload Register Map (gamma_reg.csv)")
    st.markdown("Upload your driver configuration CSV to auto-decode parameters and calculate gamma nodes.")
    reg_file = st.file_uploader("Select Register Map Data", type=["csv"], key="reg_uploader")
    
    v_gmp_parsed = 6.0
    v_gmn_parsed = -6.0
    v_gss_parsed = 0.0
    
    default_dac_str = "0, 148, 370, 513, 392, 535, 645, 465, 494, 525, 606, 434, 498, 601, 461, 641, 786, 914, 928"
    digital_gamma_vals = []
    all_params_list = []
    
    if reg_file is not None:
        try:
            content = safe_read_csv(reg_file)
            
            for idx, row in content.iterrows():
                vals = [str(x).strip() for x in row if pd.notna(x) and str(x).strip() != '']
                if not vals: continue
                
                reg_addr = vals[0].upper()
                if reg_addr.startswith("0X"):
                    hex_data = vals[1:]
                    
                    reg_info = REGISTER_MAP_DEF.get(reg_addr, {"name": "Unknown", "params": []})
                    cmd_name = reg_info.get("name", "Unknown")
                    param_defs = reg_info.get("params", [])
                    
                    if reg_addr == "0XC7":
                        dac_vals = []
                        for i in range(0, min(38, len(hex_data)-1), 2):
                            val = (int(hex_data[i].strip(), 16) << 8) | int(hex_data[i+1].strip(), 16)
                            dac_vals.append(str(val))
                        if dac_vals: 
                            default_dac_str = ", ".join(dac_vals)
                        st.success("Successfully decoded 0xC7 (19 Analog Nodes).")
                        
                    elif reg_addr == "0XC8":
                        pass 
                        
                    elif reg_addr == "0XC9":
                        pass
                        
                    elif reg_addr == "0XCF":
                        digital_gamma_vals = [int(hex_data[i].strip(), 16) for i in [4, 5, 8, 9, 10] if i < len(hex_data)]
                    
                    grouped_params = []
                    for p_idx, hex_val_str in enumerate(hex_data):
                        try:
                            dec_val = int(hex_val_str, 16)
                            hex_str = f"{dec_val:02X}"
                        except ValueError:
                            hex_str = hex_val_str.strip()
                            
                        if p_idx < len(param_defs):
                            p_name = param_defs[p_idx]
                        else:
                            p_name = f"Param_{p_idx + 1}"
                            
                        if grouped_params and grouped_params[-1]["p_name"] == p_name:
                            grouped_params[-1]["hex_str"] += hex_str
                        else:
                            grouped_params.append({
                                "Command Address": reg_addr,
                                "Command Name": cmd_name,
                                "p_name": p_name,
                                "hex_str": hex_str
                            })
                            
                    for gp in grouped_params:
                        h_str = gp["hex_str"]
                        try:
                            d_val = int(h_str, 16)
                            val_display = f"0x{h_str} (Dec: {d_val})"
                        except ValueError:
                            val_display = h_str
                            
                        all_params_list.append({
                            "Command Address": gp["Command Address"],
                            "Command Name": gp["Command Name"],
                            "Register Name": gp["p_name"],
                            "Register Value": val_display
                        })

        except Exception as e:
            st.error(f"Failed to parse register file: {e}")
            
    init_dac_input = default_dac_str

    st.markdown("---")
    st.subheader("3. Reference Voltages")
    st.markdown("Default reference voltages are configured.")
    col_v1, col_v2, col_v3 = st.columns(3)
    v_gmp_reg_val = col_v1.number_input("VGMPHO (V)", value=float(v_gmp_parsed), step=0.1)
    v_gmn_reg_val = col_v2.number_input("VGMNHO (V)", value=float(v_gmn_parsed), step=0.1)
    v_gss_reg_val = col_v3.number_input("VGS_S (V)", value=float(v_gss_parsed), step=0.1)

    st.markdown("---")
    st.subheader("4. Upload Measurement Data (gamma_sim.csv)")
    st.markdown("Upload multiple CSVs to compare datasets on the same graphs.")
    uploaded_files = st.file_uploader("Select Measurement Data", type=["csv"], accept_multiple_files=True, key="meas_uploader")
    
    st.markdown("---")
    st.header("5. Graph Settings")
    display_mode = st.radio("Luminance Mode", ("Normalized (0.0-1.0)", "Absolute (nits)"))
    is_normalized = display_mode.startswith("Norm")
    
    with st.expander("Adjust Graph Y-Axis Range", expanded=False):
        lum_min_val = -0.05 if is_normalized else -10.0
        lum_max_val = 1.05 if is_normalized else 1000.0
        y_lum_min = st.number_input("Luminance Min", value=lum_min_val, step=0.1 if is_normalized else 10.0)
        y_lum_max = st.number_input("Luminance Max", value=lum_max_val, step=0.1 if is_normalized else 10.0)
        y_gam_min = st.number_input("Gamma Min", value=1.8, step=0.1)
        y_gam_max = st.number_input("Gamma Max", value=2.6, step=0.1)
        y_duv_min = st.number_input("Δduv Min", value=-0.005, step=0.001, format="%.4f")
        y_duv_max = st.number_input("Δduv Max", value=0.020, step=0.001, format="%.4f")
        y_cct_min = st.number_input("ΔCCT Min", value=0, step=500)
        y_cct_max = st.number_input("ΔCCT Max", value=2000, step=500)
        y_li_min = st.number_input("Li Min", value=-0.05, step=0.01, format="%.2f")
        y_li_max = st.number_input("Li Max", value=0.20, step=0.01, format="%.2f")

try:
    init_dac_array = np.array([int(x.strip()) for x in init_dac_input.split(',')])
    if len(init_dac_array) != 19:
        st.sidebar.error(f"0xC7 Parse Error: Expected 19 values but got {len(init_dac_array)}.")
        st.stop()
except ValueError:
    st.sidebar.error("Failed to parse initial DAC values from uploaded CSV.")
    st.stop()

if all_params_list:
    st.subheader("📥 All Command Register Parameters")
    st.dataframe(pd.DataFrame(all_params_list), width="stretch", hide_index=True)
    st.markdown("---")

# ==========================================
# File Processing
# ==========================================
meas_datasets = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file, header=None, names=[0, 1, 2, 3], engine='python')
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            if len(df.columns) >= 4:
                df_clean = df_numeric.dropna(subset=[0, 1, 2, 3])
                if not df_clean.empty:
                    meas_datasets.append({
                        "name": uploaded_file.name,
                        "meas_gray": df_clean.iloc[:, 0].values,
                        "meas_x": df_clean.iloc[:, 1].values,
                        "meas_y": df_clean.iloc[:, 2].values,
                        "meas_lum": df_clean.iloc[:, 3].values
                    })
        except Exception as e:
            st.error(f"Failed to load CSV {uploaded_file.name}: {e}")
else:
    st.info("👈 Please upload Measurement Data CSV files from the sidebar. Currently displaying demo data.")
    demo_gray = np.linspace(255, 0, 256)
    meas_datasets.append({
        "name": "Demo Data",
        "meas_gray": demo_gray,
        "meas_lum": (demo_gray / 255.0) ** 2.5 * 500.0,
        "meas_x": np.full_like(demo_gray, 0.3127),
        "meas_y": np.full_like(demo_gray, 0.3290)
    })

if meas_datasets:
    
    st.header("Tuning Curves & Voltage Tracking")
    
    dataset_names = [d["name"] for d in meas_datasets]
    selected_datasets = st.multiselect(
        "📊 Select Datasets to Display", 
        options=dataset_names, 
        default=dataset_names
    )
    
    st.markdown("""
    **【Voltage Graph Guide】**
    * 🔴🔵 **Solid Line/Markers (Input Table Data):** Hardware output voltage calculated directly from the input register values. Matches the table below.
    """)
    
    col_t2, col_t3 = st.columns(2)
    show_target = col_t2.checkbox("Target Curve", value=True)
    show_adj = col_t3.checkbox("Adjusted Curve (TD7875)", value=True)
    
    col_b1, col_b2 = st.columns(2)
    show_meas_cp = col_b1.checkbox("Measured Points", value=True)
    show_cp = col_b2.checkbox("TD7875 CP Points (19 Nodes)", value=True) 
    
    fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=(
            "Grayscale vs Luminance", "Grayscale vs Gamma Value", 
            "Grayscale vs Hardware Output Voltage (Volts)", "Relative Luminance Diff: Li = (Ln+1 - Ln) / Ln",
            "Grayscale vs Δu'v' (Δduv)", "Grayscale vs ΔCCT"
        ),
        vertical_spacing=0.1
    )
    
    PLOT_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    dataset_results = []
    
    x_cont_global = np.linspace(0, 255, 256)
    cp_names_global = [f"V{int(x)}P/N" for x in X_POINTS]
    
    v_pos_init_cp_global = convert_dac_to_physical_voltage(init_dac_array, 'positive', v_gmp_reg_val, v_gmn_reg_val, v_gss_reg_val)
    v_pos_full_init_global = apply_td7875_hardware_formulas(dict(zip(X_POINTS, v_pos_init_cp_global)))
    
    v_neg_init_cp_global = convert_dac_to_physical_voltage(init_dac_array, 'negative', v_gmp_reg_val, v_gmn_reg_val, v_gss_reg_val)
    v_neg_full_init_global = apply_td7875_hardware_formulas(dict(zip(X_POINTS, v_neg_init_cp_global)))

    fig.add_trace(go.Scatter(
        x=x_cont_global, y=v_pos_full_init_global, mode='lines', 
        name="Input Pos Voltage (Table Data)", line=dict(color='red', width=2),
        hovertemplate="Gray: %{x}<br>Pos Vol: %{y:.3f} V<extra></extra>"
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_cont_global, y=v_neg_full_init_global, mode='lines', 
        name="Input Neg Voltage (Table Data)", line=dict(color='blue', width=2),
        hovertemplate="Gray: %{x}<br>Neg Vol: %{y:.3f} V<extra></extra>"
    ), row=2, col=1)
    
    if show_cp:
        hover_text_pos = [f"Register: VGMP{i}<br>Gray: {int(X_POINTS[i])}<br>Value: 0x{init_dac_array[i]:02X} (Dec: {init_dac_array[i]})<br>Pos Voltage: {v_pos_init_cp_global[i]:.3f} V" for i in range(len(X_POINTS))]
        fig.add_trace(go.Scatter(
            x=X_POINTS, y=v_pos_init_cp_global, mode='markers',
            marker=dict(color='red', size=8, symbol='circle', line=dict(color='black', width=1)), showlegend=False,
            text=hover_text_pos, hovertemplate="%{text}<extra></extra>"
        ), row=2, col=1)
        
        hover_text_neg = [f"Register: VGMN{i}<br>Gray: {int(X_POINTS[i])}<br>Value: 0x{init_dac_array[i]:02X} (Dec: {init_dac_array[i]})<br>Neg Voltage: {v_neg_init_cp_global[i]:.3f} V" for i in range(len(X_POINTS))]
        fig.add_trace(go.Scatter(
            x=X_POINTS, y=v_neg_init_cp_global, mode='markers',
            marker=dict(color='blue', size=8, symbol='circle', line=dict(color='black', width=1)), showlegend=False,
            text=hover_text_neg, hovertemplate="%{text}<extra></extra>"
        ), row=2, col=1)

    for idx, data in enumerate(meas_datasets):
        meas_gray = data["meas_gray"]
        meas_lum = data["meas_lum"]
        meas_x = data["meas_x"]
        meas_y = data["meas_y"]
        name = data["name"]

        res = process_td7875_physical_tuning(meas_gray, meas_lum, init_dac_array, v_gmp_reg_val, v_gmn_reg_val, v_gss_reg_val, target_gamma_input)
        
        denom = -2 * meas_x + 12 * meas_y + 3
        valid_xy = denom != 0
        u_prime = np.zeros_like(meas_x)
        v_prime = np.zeros_like(meas_y)
        u_prime[valid_xy] = (4 * meas_x[valid_xy]) / denom[valid_xy]
        v_prime[valid_xy] = (9 * meas_y[valid_xy]) / denom[valid_xy]

        n = (meas_x - 0.3320) / (0.1858 - meas_y)
        cct = 449 * (n**3) + 3525 * (n**2) + 6823.3 * n + 5520.33

        ref_idx = np.argmax(meas_gray) 
        u_ref = u_prime[ref_idx]
        v_ref = v_prime[ref_idx]
        cct_ref = cct[ref_idx]

        delta_cct = np.sqrt((cct - cct_ref)**2)
        delta_uv = np.sqrt((u_prime - u_ref)**2 + (v_prime - v_ref)**2)

        l_max = np.max(meas_lum)
        gray_norm = meas_gray / 255.0
        calc_gamma = np.zeros_like(meas_gray, dtype=float)
        valid_lum = (meas_gray > 0) & (meas_gray < 255) & (meas_lum > 0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            calc_gamma[valid_lum] = np.log(meas_lum[valid_lum] / l_max) / np.log(gray_norm[valid_lum])
        calc_gamma[~valid_lum] = np.nan
        
        sort_asc = np.argsort(meas_gray)
        lum_asc = meas_lum[sort_asc]
        
        l_diff_asc = np.zeros_like(lum_asc)
        l_diff_asc[1:] = np.diff(lum_asc)
        l_diff_asc[0] = np.nan
        l_diff = np.zeros_like(meas_lum)
        l_diff[sort_asc] = l_diff_asc

        l_diff_rel_asc = np.zeros_like(lum_asc)
        with np.errstate(divide='ignore', invalid='ignore'):
            l_diff_rel_asc[1:] = np.diff(lum_asc) / lum_asc[:-1]
        l_diff_rel_asc[0] = np.nan
        l_diff_rel = np.zeros_like(meas_lum)
        l_diff_rel[sort_asc] = l_diff_rel_asc

        dataset_results.append({
            "name": name, "res": res, "meas_gray": meas_gray, "meas_lum": meas_lum,
            "meas_x": meas_x, "meas_y": meas_y, "u_prime": u_prime, "v_prime": v_prime,
            "cct": cct, "delta_cct": delta_cct, "delta_uv": delta_uv,
            "calc_gamma": calc_gamma, "l_diff": l_diff, "l_diff_rel": l_diff_rel
        })

        if name in selected_datasets:
            scale_div = res['max_lum'] if is_normalized else 1.0
            color = PLOT_COLORS[idx % len(PLOT_COLORS)]
            
            # Row 1 Col 1
            if show_target:
                fig.add_trace(go.Scatter(x=res['x_cont'], y=res['lum_tgt']/scale_div, name=f"Target ({name})", line=dict(color=color, width=2)), row=1, col=1)
            if show_adj:
                fig.add_trace(go.Scatter(x=res['x_cont'], y=res['lum_adj']/scale_div, name=f"Adj ({name})", line=dict(dash='dot', color=color, width=2)), row=1, col=1)

            if show_meas_cp:
                fig.add_trace(go.Scatter(x=meas_gray, y=meas_lum / scale_div, mode='markers', name=f"Points ({name})", marker=dict(color=color, size=5, symbol='circle-open', line=dict(width=1.5)), hovertemplate=f"[{name}]<br>Gray: %{{x}}<br>Lum: %{{y:.4f}}<extra></extra>"), row=1, col=1)

            if show_cp:
                hover_text = [f"[{name}] VGMP{i}/VGMN{i}<br>Gray: {int(res['cp_x'][i])}<br>After: 0x{res['adj_dac'][i]:02X}" for i in range(len(res['cp_x']))]
                fig.add_trace(go.Scatter(x=res['cp_x'], y=np.interp(res['cp_x'], res['x_cont'], res['lum_adj']) / scale_div, mode='markers', name=f"CP ({name})", marker=dict(color=color, size=7, symbol='diamond'), text=hover_text, hovertemplate="%{text}<br>Lum: %{y:.4f}<extra></extra>"), row=1, col=1)

            # Row 1 Col 2
            if show_target:
                fig.add_trace(go.Scatter(x=res['x_cont'], y=res['gam_tgt'], name=f"Target Gamma ({name})", line=dict(color=color, width=2), showlegend=False), row=1, col=2)
            if show_adj:
                fig.add_trace(go.Scatter(x=res['x_cont'], y=res['gam_adj'], name=f"Adj Gamma ({name})", line=dict(dash='dot', color=color, width=2), showlegend=False), row=1, col=2)

            if show_meas_cp:
                fig.add_trace(go.Scatter(x=meas_gray, y=calc_gamma, mode='markers', name=f"Gamma Points ({name})", marker=dict(color=color, size=5, symbol='circle-open', line=dict(width=1.5)), hovertemplate=f"[{name}]<br>Gray: %{{x}}<br>Gamma: %{{y:.3f}}<extra></extra>", showlegend=False), row=1, col=2)

            if show_cp:
                fig.add_trace(go.Scatter(x=res['cp_x'], y=np.interp(res['cp_x'], res['x_cont'], res['gam_adj']), mode='markers', name=f"CP Gamma ({name})", marker=dict(color=color, size=7, symbol='diamond'), text=hover_text, hovertemplate="%{text}<br>Gamma: %{y:.3f}<extra></extra>", showlegend=False), row=1, col=2)

            # Row 2 Col 2: Li
            li_tgt = np.zeros(256)
            li_adj = np.zeros(256)
            with np.errstate(divide='ignore', invalid='ignore'):
                li_tgt[1:] = np.diff(res['lum_tgt']) / res['lum_tgt'][:-1]
                li_adj[1:] = np.diff(res['lum_adj']) / res['lum_adj'][:-1]
                
            li_tgt[0] = li_adj[0] = np.nan

            if show_target:
                fig.add_trace(go.Scatter(x=res['x_cont'], y=li_tgt, name=f"Target Li ({name})", line=dict(color=color, width=2), showlegend=False), row=2, col=2)
            if show_adj:
                fig.add_trace(go.Scatter(x=res['x_cont'], y=li_adj, name=f"Adj Li ({name})", line=dict(dash='dot', color=color, width=2), showlegend=False), row=2, col=2)

            # Row 3: uv, CCT
            fig.add_trace(go.Scatter(x=meas_gray, y=delta_uv, mode='lines+markers', name=f"Δduv ({name})", line=dict(color=color, width=2), marker=dict(size=4)), row=3, col=1)
            fig.add_trace(go.Scatter(x=meas_gray, y=delta_cct, mode='lines+markers', name=f"ΔCCT ({name})", line=dict(color=color, width=2), marker=dict(size=4)), row=3, col=2)

    fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=2, col=2, annotation_text="Criteria: Li < 0.05", annotation_position="top right")

    y_title = 'Normalized Luminance' if is_normalized else 'Luminance (nits)'
    fig.update_xaxes(title_text="Input Grayscale (0-255)", range=[-5, 260])
    fig.update_yaxes(title_text=y_title, row=1, col=1, range=[y_lum_min, y_lum_max])
    fig.update_yaxes(title_text="Gamma Value", row=1, col=2, range=[y_gam_min, y_gam_max])
    fig.update_yaxes(title_text="Output Voltage (V)", row=2, col=1)
    fig.update_yaxes(title_text="Li", row=2, col=2, range=[y_li_min, y_li_max])
    fig.update_yaxes(title_text="Δu'v' (Δduv)", row=3, col=1, range=[y_duv_min, y_duv_max])
    fig.update_yaxes(title_text="ΔCCT (K)", row=3, col=2, range=[y_cct_min, y_cct_max])
    
    fig.update_layout(
        height=1300, 
        hovermode="closest", 
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="lightgray", borderwidth=1), 
        margin=dict(t=80, b=150)
    )
    st.plotly_chart(fig, width="stretch")
    st.markdown("---")

    # ==========================================
    # Table Display
    # ==========================================
    st.header("LCD Gamma Output Parameters & Data")
    
    tabs = st.tabs([d["name"] for d in dataset_results])
    
    for idx, tab in enumerate(tabs):
        d_res = dataset_results[idx]
        res = d_res["res"]
        
        with tab:
            col_reg1, col_reg2 = st.columns([2, 1])
            with col_reg1:
                st.subheader("Analog Gamma Registers (0xC7)")
                st.caption("Calculated hardware physical gamma voltages based on input register values.")
                
                col_pos, col_neg = st.columns(2)

                td7875_pos = []
                td7875_neg = []
                for i in range(len(res['cp_x'])):
                    td7875_pos.append({
                        "Node": f"V{int(res['cp_x'][i])}P",
                        "Reg Name": f"VGMP{i}",
                        "Value": f"0x{res['init_dac'][i]:02X} ({res['init_dac'][i]})",
                        "Voltage": f"{res['v_pos_init_cp'][i]:.3f} V"
                    })
                    td7875_neg.append({
                        "Node": f"V{int(res['cp_x'][i])}N",
                        "Reg Name": f"VGMN{i}",
                        "Value": f"0x{res['init_dac'][i]:02X} ({res['init_dac'][i]})",
                        "Voltage": f"{res['v_neg_init_cp'][i]:.3f} V"
                    })

                with col_pos:
                    st.markdown("**🔴 Positive**")
                    st.dataframe(pd.DataFrame(td7875_pos), height=680, hide_index=True, width="stretch")

                with col_neg:
                    st.markdown("**🔵 Negative**")
                    st.dataframe(pd.DataFrame(td7875_neg), height=680, hide_index=True, width="stretch")
                
            with col_reg2:
                st.subheader("Reference Voltages")
                st.info(f"**VGMPHO:** {v_gmp_reg_val} V \n\n **VGMNHO:** {v_gmn_reg_val} V \n\n **VGS_S:** {v_gss_reg_val} V")

            st.markdown("---")
            st.subheader("Detailed Measurement Data")

            detailed_df = pd.DataFrame({
                "Gray": d_res["meas_gray"], "x": d_res["meas_x"], "y": d_res["meas_y"], "L (nits)": d_res["meas_lum"],
                "u'": d_res["u_prime"], "v'": d_res["v_prime"], "Gamma": d_res["calc_gamma"], 
                "L diff": d_res["l_diff"], "Li (Rel Diff)": d_res["l_diff_rel"],
                "CCT (K)": d_res["cct"], "ΔCCT": d_res["delta_cct"], "Δu'v' (Δduv)": d_res["delta_uv"]
            })

            def style_df(row):
                styles = [''] * len(row)
                try:
                    ldiff_idx = row.index.get_loc("L diff")
                    li_idx = row.index.get_loc("Li (Rel Diff)")
                    if pd.notna(row["L diff"]) and row["L diff"] < 0:
                        styles[ldiff_idx] = 'color: red; font-weight: bold;'
                    li_val = row["Li (Rel Diff)"]
                    if pd.notna(li_val) and li_val < 0:
                        styles[li_idx] = 'color: red; font-weight: bold;' 
                except Exception:
                    pass
                return styles

            styled_df = detailed_df.style.apply(style_df, axis=1).format({
                "Gray": "{:.0f}", "x": "{:.4f}", "y": "{:.4f}", "L (nits)": "{:.2f}",
                "u'": "{:.4f}", "v'": "{:.4f}", "Gamma": "{:.3f}", "L diff": "{:.3f}",
                "Li (Rel Diff)": "{:.4f}", "CCT (K)": "{:.0f}", "ΔCCT": "{:.0f}", "Δu'v' (Δduv)": "{:.4f}"
            }, na_rep="-")

            st.dataframe(styled_df, width="stretch", height=500)