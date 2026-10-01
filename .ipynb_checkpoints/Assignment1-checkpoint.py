import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import numpy as np

campaign_data = pd.read_csv('skin clinic campaign.csv')

print("=====================================")
print("              Head                   ")
print("=====================================")
print(campaign_data.head())
print("=====================================")
print("              Tail                   ")
print("=====================================")
print(campaign_data.tail())
print("=====================================")
print("              Info                   ")
print("=====================================")
print(campaign_data.info())
print("=====================================")
print("              Describe               ")
print("=====================================")
print(campaign_data.describe())
print("=====================================")
print("              DTypes                 ")
print("=====================================")
print(campaign_data.dtypes)

# ================================
# CATEGORICAL
# ================================
cat_cols = ['AgeGroup', 'Gender', 'Purchase_Last_Quarter', 'Unique_Products_Purchased']
masterdata = campaign_data

for col in cat_cols:
    masterdata[col] = masterdata[col].astype('category')
    campaign_data[col] = campaign_data[col].astype('category')

# ================================
# DUMMIES
# ================================
master_dum = pd.get_dummies(masterdata, columns=cat_cols, drop_first=True)
campaign_dum = pd.get_dummies(campaign_data, columns=cat_cols, drop_first=True)

campaign_dum = campaign_dum.reindex(columns=master_dum.columns, fill_value=0)


age_order = ['<30', '30–50', '>50']
if pd.api.types.is_numeric_dtype(campaign_data['AgeGroup']):
    campaign_data['Age_Band'] = pd.cut(campaign_data['AgeGroup'],
                                       bins=[-np.inf, 29, 50, np.inf], labels=age_order)
else:
    campaign_data['Age_Band'] = campaign_data['AgeGroup']   # already banded in the file
    age_order = None
    

# ================================
# MODEL
# ================================
y = masterdata['Response_to_Campaign']
if y.dtype == object:
    y = y.map({'Yes': 1, 'No': 0})
y = y.astype(float)
campaign_data['Responded'] = y


X = master_dum.drop(columns=['Response_to_Campaign', 'CustID'])

X = sm.add_constant(X)

X = X.astype(float)
y = y.astype(float)

model = sm.Logit(y, X).fit(disp=False)

# ================================
# COEFFICIENT TABLE
# ================================
coef_df = pd.DataFrame({
    "Feature": model.params.index,
    "Coefficient": model.params.values,
    "P-Value": model.pvalues.values
}).round(3)

# ================================
# PIE CHART
# ================================
response_counts = masterdata['Response_to_Campaign'].value_counts().reset_index()
response_counts.columns = ['Response_to_Campaign', 'Count']

pie_fig = px.pie(
    response_counts,
    names='Response_to_Campaign',
    values='Count',
    hole=0.4
)

# ================================
# PREDICTIONS (THRESHOLD 0.13)
# ================================
X_test = campaign_dum.drop(columns=['Response_to_Campaign', 'CustID'])
X_test = sm.add_constant(X_test)
X_test = X_test.astype(float)

campaign_data['Probability'] = model.predict(X_test)
campaign_data['Predicted_Class'] = (campaign_data['Probability'] > 0.13).astype(int)

prediction_table = campaign_data[['CustID', 'Probability', 'Predicted_Class']].copy()
prediction_table['Probability'] = prediction_table['Probability'].round(3)



def response_rate(data, group_col, order=None, label=None):
    table = (data.groupby(group_col, observed=True)['Responded']
                 .agg(Customers='count', Responders='sum'))
    table['Response_Rate_%'] = (table['Responders'] / table['Customers'] * 100).round(2)
    if order is not None:
        table = table.reindex(order)

    table = table.reset_index().rename(columns={group_col: 'Segment'})
    table['Segment'] = table['Segment'].astype(str)
    
     # First column: 'Age_Band' -> 'Age', 'Gender' -> 'Gender', or use label 
    table.insert(0, 'Response_Segmentation', label or group_col.replace('_Band', ''))
    return table

gender_table   = response_rate(campaign_data, 'Gender')
print(gender_table)
age_table      = response_rate(campaign_data, 'Age_Band', order=age_order)
print(age_table)
product_order = ['1–4', '5–8', '>8']
campaign_data['Product_Band'] = pd.cut(campaign_data['Unique_Products_Purchased'],
                                       bins=[0, 4, 8, np.inf], labels=product_order)
product_table  = response_rate(campaign_data, 'Product_Band', order=product_order)
print(product_table)

def generate_campaign_analysis_summary():
    print("called generate_campaign_analysis_summary()")
    df = pd.concat([
        response_rate(campaign_data, 'Gender'),
        response_rate(campaign_data, 'Age_Band', order=age_order),
        response_rate(campaign_data, 'Purchase_Last_Quarter', order=['Yes', 'No']),
        response_rate(campaign_data, 'Product_Band', order=product_order),
    ], ignore_index=True)    
    summary = df.round(4).to_dict(orient="records") 
    print(f"returning from generate_campaign_analysis_summary() {summary}")
    return summary 

df = generate_campaign_analysis_summary()
df


##############################################################
# The Fast-PAI applicaiton
###############################################################

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# -------------------------------------------------
# Create FastAPI app
# -------------------------------------------------
app = FastAPI()



# -------------------------------------------------
# API endpoint returning JSON data
# -------------------------------------------------
@app.get("/campaign-analysis")
def get_campaign_analysis_summary():
    df  = generate_campaign_analysis_summary()
    return df


# -------------------------------------------------
# HTML frontend embedded directly in endpoint
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Skin Care Campaign Summary</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
            }
            button {
                padding: 10px 16px;
                font-size: 16px;
                margin-bottom: 20px;
                cursor: pointer;
            }
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #ccc;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #f4f4f4;
            }
        </style>
    </head>
    <body>

        <h2>Skin Care Campaign Summary</h2>

        <button onclick="loadData()">Load Summary</button>

        <table id="summaryTable">
            <thead>
                <tr>
                    <th>Segment Category</th>
                    <th>Segment</th>                    
                    <th>Customers</th>
                    <th>Responders</th>
                    <th>Response Rate</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>

        <script>
            function loadData() {
                fetch('/campaign-analysis')
                    .then(response => response.json())
                    .then(data => {
                        const tbody = document.querySelector('#summaryTable tbody');
                        tbody.innerHTML = '';

                        data.forEach(row => {
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td>${row.Response_Segmentation}</td>
                                <td>${row.Segment}</td>
                                <td>${row.Customers}</td>
                                <td>${row.Responders}</td>
                                <td>${row["Response_Rate_%"]}%</td>
                            `;
                            tbody.appendChild(tr);
                        });
                    })
                    .catch(error => {
                        alert('Error fetching data');
                        console.error(error);
                    });
            }
        </script>

    </body>
    </html>
    """

    return html_content