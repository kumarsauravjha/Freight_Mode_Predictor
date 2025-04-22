import gradio as gr
import pandas as pd
import numpy as np
import joblib
import pickle
import json

# --- Load all saved artifacts ---
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")
enc_hs = joblib.load("enc_hs.pkl")
enc_ship = joblib.load("enc_ship.pkl")
enc_route = joblib.load("enc_route.pkl")

with open("shiptype_lookup.pkl", "rb") as f:
    shiptype_lookup = pickle.load(f)

with open("distance_lookup.pkl", "rb") as f:
    distance_lookup = pickle.load(f)

with open("country-centroids.json", "r") as f:
    centroids = json.load(f)
iso_centroids = {c["alpha3"]: (c["latitude"], c["longitude"]) for c in centroids}

with open("modes_by_origin.pkl","rb") as f:
    modes_by_origin = pickle.load(f)


ALL_ISO = ['JPN', 'MYS', 'TWN', 'TJK', 'UKR', 'MEX', 'MDA', 'NIC', 'NGA',
       'PAK', 'PAN', 'PRY', 'PER', 'PHL', 'AUS', 'AUT', 'AZE', 'BGD',
       'BEL', 'BOL', 'BRA', 'BGR', 'KHM', 'CMR', 'CAN', 'CHN', 'CZE',
       'DNK', 'DOM', 'ECU', 'EGY', 'ETH', 'FIN', 'FRA', 'GIN', 'HTI',
       'HND', 'HKG', 'HUN', 'IND', 'BLR', 'COL', 'DEU', 'IDN', 'IRN',
       'ITA', 'MNG', 'MAR', 'MOZ', 'MMR', 'NLD', 'NZL', 'AFG', 'DZA',
       'AGO', 'ARG', 'ARM', 'TCD', 'CHL', 'CIV', 'HRV', 'CUB', 'GEO',
       'GHA', 'GRC', 'GTM', 'JAM', 'JOR', 'KAZ', 'KEN', 'RUS', 'SAU',
       'SEN', 'ESP', 'LKA', 'SDN', 'SWE', 'SYR', 'TZA', 'GBR', 'USA',
       'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE', 'MKD', 'MNE', 'SVK',
       'SVN', 'TKM', 'MDG', 'ISL', 'ARE', 'IRQ', 'IRL', 'LVA', 'LBN',
       'LTU', 'MLI', 'POL', 'PRT', 'COG', 'ROU', 'SRB', 'SGP', 'SOM',
       'ZAF', 'KOR', 'THA', 'TUN', 'TUR', 'UGA', 'URY', 'NOR', 'ALB',
       'BIH', 'CHE', 'CYP', 'EST', 'KGZ', 'MLT', 'BRN', 'CAF', 'COD',
       'PNG', 'NAM', 'LAO', 'PRI', 'QAT', 'BHR', 'MWI', 'NPL', 'TGO',
       'BEN', 'AIA', 'MSR', 'TCA', 'VCT', 'ERI', 'STP', 'BLZ', 'BTN',
       'BDI', 'KIR', 'NRU', 'NIU', 'PLW', 'WSM', 'TUV', 'VUT', 'MDV',
       'TLS', 'CPV', 'MNP', 'FSM', 'GNB', 'MRT', 'NER', 'SLE', 'GUY',
       'LUX', 'BFA', 'RWA', 'AND', 'BWA', 'MAC', 'SLV', 'SWZ', 'FLK',
       'GAB', 'ISR', 'PRK', 'LBY', 'NFK', 'OMN', 'MAF', 'SPM', 'SMR',
       'SSD', 'PSE', 'ESH', 'ABW', 'DJI', 'LSO', 'LBR', 'CRI', 'GIB',
       'KWT', 'TTO', 'GMB', 'SUR', 'BES', 'CUW', 'GNQ', 'BLM', 'SXM',
       'SLB', 'MUS', 'VGB', 'BRB', 'KNA', 'LCA', 'VIR', 'BHS', 'BMU',
       'COM', 'PYF', 'GRD', 'SYC', 'CYM', 'FJI', 'MHL', 'NCL', 'TON',
       'ASM', 'ATG', 'GLP', 'COK', 'DMA', 'CXR', 'FRO', 'GRL', 'GUM',
       'SHN', 'TKL']

HS_CHOICES = sorted([
    "Chemicals_plastic","Paper_wood","Electronic_devices","Livestock","Food",
    "Other_metals","Transport_equipment","Other_minerals","Iron_steel","Textile",
    "Other_agriculture","Metal_products","Other_mining","Rice_crops",
    "Other_manufacturing","Refined_oil","Crude oil","Gas","Coal"
])

# --- Predict function using lookups + encoders ---
def predict_mode(user_input_df):
    user_input_df['origin_destination'] = user_input_df['origin_ISO'] + "_" + user_input_df['destination_ISO']
    user_input_df['ship_type'] = user_input_df.apply(
        lambda row: shiptype_lookup.get((row['origin_ISO'], row['destination_ISO']), 'Container'), axis=1
    )
    user_input_df['distance(km)'] = user_input_df.apply(
        lambda row: distance_lookup.get((row['origin_ISO'], row['destination_ISO']), np.nan), axis=1
    )

    user_input_df['IFM_HS_encoded'] = enc_hs.transform(user_input_df[['IFM_HS']])
    user_input_df['ship_type_encoded'] = enc_ship.transform(user_input_df[['ship_type']])
    user_input_df['origin_destination_encoded'] = enc_route.transform(user_input_df[['origin_destination']])

    user_input_final = pd.DataFrame([[
        user_input_df['distance(km)'].values[0],
        user_input_df['flow(tonne)'].values[0],
        user_input_df['origin_destination_encoded'].values[0],
        user_input_df['ship_type_encoded'].values[0],
        user_input_df['IFM_HS_encoded'].values[0]
    ]], columns=['distance(km)', 'flow(tonne)', 'origin_destination_encoded', 'ship_type_encoded', 'IFM_HS_encoded'])

    # Predict probabilities
    probs = model.predict_proba(user_input_final)[0]
    class_names = le.inverse_transform(np.arange(len(probs)))
    top_class = class_names[np.argmax(probs)]

    return {
        "label": str(top_class),
        "confidences": {str(cls): float(prob) for cls, prob in zip(class_names, probs)}
    }


def predict_mode_web(origin_ISO, destination_ISO, flow, IFM_HS):
    try:
        # 1) building the df and getting the raw model output
        df = pd.DataFrame([{
            "origin_ISO": origin_ISO.strip().upper(),
            "destination_ISO": destination_ISO.strip().upper(),
            "flow(tonne)": flow,
            "IFM_HS": IFM_HS.strip()
        }])
        result = predict_mode(df)
        confs = result["confidences"]
        sorted_modes = sorted(confs.items(), key=lambda kv: kv[1], reverse=True)
       

        # 2) pull the set of “known” modes for this origin
        known = modes_by_origin.get(origin_ISO.upper(), set())

        # 3) build the markdown
        lines = ["**Recommended mode of transport for your shipment:**"]
        for mode, prob in sorted_modes:
            pct = f"{prob*100:.0f}%"
            if mode not in known:
                # tag it as “new MoT”
                lines.append(f"- **{mode}** ({pct})  ← _new MoT!_")
            else:
                lines.append(f"- {mode} ({pct})")

        return "\n\n".join(lines)

    except Exception:
        return " **Error**: something went wrong with your inputs."


# --- Wrap for Gradio ---
# def predict_mode_web(origin_ISO, destination_ISO, flow, IFM_HS):
#     try:
#         df = pd.DataFrame([{
#             "origin_ISO": origin_ISO.strip().upper(),
#             "destination_ISO": destination_ISO.strip().upper(),
#             "flow(tonne)": flow,
#             "IFM_HS": IFM_HS.strip()
#         }])
#         return predict_mode(df)

#     except Exception as e:
#         return {
#             "label": "Error",
#             "confidences": {
#                 "Air": 0.0,
#                 "Rail": 0.0,
#                 "Road": 0.0,
#                 "Sea": 0.0,
#                 "Error": 1.0
#             }
#         }

# # --- Gradio UI ---
# iface = gr.Interface(
#     fn=predict_mode_web,
#     inputs=[
#         gr.Dropdown(
#             label="🌍 Origin ISO", 
#             choices=ALL_ISO,
#             value="KEN",            # default value if you like
#             elem_id="input-origin"
#         ),
#         gr.Dropdown(
#             label="🎯 Destination ISO", 
#             choices=ALL_ISO,
#             value="UGA",
#             elem_id="input-dest"
#         ),
#         gr.Number(label="⚖️ Flow (tonne)", value=0.06, elem_id="input-num"),
#         gr.Dropdown(label="📦 HS category", choices=HS_CHOICES, value=HS_CHOICES[0])
#     ],
#     outputs=gr.JSON(label="📊 Mode Probabilities"),
#     title="🌐 Global Freight Mode Predictor",
#     description="🔍 Enter origin, destination, flow and HS code to predict the transport mode probabilities.",
#     theme="default",
#     css="""
#         /* 1) Input labels */
#         .gradio-container .input_block .component-title {
#           font-size: 22px !important;
#           font-weight: 600 !important;
#         }

#         /* 2) JSON output label */
#         .gradio-container .output_block .component-title {
#           font-size: 22px !important;
#           font-weight: 600 !important;
#         }

#         /* 3) Page title & desc */
#         .prose h1 {
#           font-size: 32px !important;
#           font-weight: bold !important;
#         }
#         .prose p {
#           font-size: 18px !important;
#         }

#         /* 4) Field heights */
#         #input-text input, #input-num input {
#             font-size: 22px;
#             height: 48px;
#         }
#     """
# )
custom_css = """
/* Input + Output labels */
.big-label * {
    font-size: 20px !important;
}

/* Markdown output label */
.output_block .component-title {
    font-size: 22px !important;
    font-weight: 600 !important;
}

/* Page description paragraphs */
.prose p {
    font-size: 30px !important;
    font-weight: 500 !important;
    color: #d1d1d1 !important;
}

/* Your custom title */
/* you generated an <h1 id="main-title">, so just match #main-title directly */
#main-title {
    font-size: 48px !important;
    font-weight: 900 !important;
    color: white !important;
    text-align: center !important;
    margin-bottom: 20px !important;
}
"""


# --- Gradio UI with custom CSS ---
with gr.Blocks(css=custom_css) as iface:
    gr.HTML('<h1 id="main-title">🌐 Global Freight Mode Predictor</h1>')
    gr.Markdown("🔍 Select Origin, Destination, Weight and Commodity Type to predict the most suitable transport mode.")

    origin = gr.Dropdown(label="🌍 Origin country (ISO Code)", choices=ALL_ISO, value="KEN", elem_classes="big-label")
    destination = gr.Dropdown(label="🎯 Destination country (ISO Code)", choices=ALL_ISO, value="UGA", elem_classes="big-label")
    flow = gr.Number(label="⚖️ Weight of Shipment (in tonnes)", value=0.06, elem_classes="big-label")
    hs_code = gr.Dropdown(label="📦 Type of Commodity", choices=HS_CHOICES, value=HS_CHOICES[0], elem_classes="big-label")

    output = gr.Markdown(label="📊 Recommended Mode of Transport for your shipment:")

    btn = gr.Button("🚀 Predict", elem_classes="big-label")
    btn.click(fn=predict_mode_web,
              inputs=[origin, destination, flow, hs_code],
              outputs=[output])

iface.launch()
# if __name__ == "__main__":
#     iface.launch()


