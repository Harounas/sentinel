import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import dateutil.parser as parser
import altair as alt
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.dates import DateFormatter
import ssl
import requests
from io import StringIO
import seaborn as sns
ssl._create_default_https_context = ssl._create_stdlib_context
from datetime import datetime, timedelta
import io
#from st_vizzu import *
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats import zscore
from column_name import replacements
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression, RFE
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.utils import check_array
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.sandbox.stats.multicomp import multipletests

# Define the URL for Google Sheets
url = "https://docs.google.com/spreadsheets/d/1mMqW_iRtzPG26nl10dGxLwZXry8I49u0NBrMOekiULY/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)

# Title and markdown for the app
st.title("Data Summary Sentinel Project")
st.markdown("---")

# Function to load data from Google Sheets
@st.cache_data
def load_data(datapath):
    dataset = conn.read(spreadsheet=datapath)
    return dataset

# Text input widget for token
token = st.sidebar.text_input("Input a token")

# Function to process uploaded files or fetch data using token
def dataprocess(uploaded_file):
    if token:
        # If a token is provided, fetch data using the token from the API
        data = {
            'token': token,
            'content': 'record',
            'action': 'export',
            'format': 'csv',
            'type': 'flat',
            'csvDelimiter': '',
            'fields[0]': 'participantid_site',
            'forms[0]': 'case_report_form',
            'forms[1]': 'sample_collection_form',
            'forms[2]': 'rdt_laboratory_report_form',
            'forms[3]': 'pcr_laboratory_report_form',
            'forms[4]': 'urinalysis_laboratory_report_form',
            'forms[5]': 'malaria_laboratory_report_form',
            'rawOrLabel': 'raw',
            'rawOrLabelHeaders': 'raw',
            'exportCheckboxLabel': 'false',
            'exportSurveyFields': 'false',
            'exportDataAccessGroups': 'false',
            'returnFormat': 'csv'
        }
        r = requests.post('https://redcap-acegid.org/api/', data=data)
        df = pd.read_csv(StringIO(r.text), low_memory=False)
        st.write(df.head())
        return df
    elif uploaded_file is not None:
        # If files are uploaded, merge them
        dfS = []
        for uploaded_f in uploaded_file:
            d = pd.read_csv(uploaded_f)
            dfS.append(d)

        if dfS:
            dfmerge = pd.concat(dfS, ignore_index=True)
            df0 = dfmerge.drop_duplicates()
            st.write("Merged DataFrame:")
            st.write(df0)

            # Optionally: Save merged DataFrame as a new file
            st.download_button(
                label="Download Merged CSV",
                data=df0.to_csv(index=False),
                file_name="merged_dataframe.csv",
                mime="text/csv"
            )
            return df0
        else:
            st.write("No files uploaded or files are empty.")
            return load_data(url)
    else:
        # If neither files nor token is provided, load data from Google Sheets
        st.write("No file uploaded or token provided.")
        return load_data(url)

# Main function to handle file uploads or token input
def main():
    st.title('File Upload or Provide Token')
    uploaded_file = st.sidebar.file_uploader("Upload a file", accept_multiple_files=True, type="csv")
    df = dataprocess(uploaded_file)
    return df

# Execute the main function
df = main()

   
if 'siteregion_crf' not in df.columns:
    # Rename 'City/Village' to 'siteregion_crf'
    df = df.rename(columns={'site_sample':'siteregion_crf'})
    df['siteregion_crf']=df['siteregion_crf'].replace({1: "KGH",2: "Kailahun GH",3: "Bo GH", 4: "CHPRL", 5: "Community",6: "Other government hospital",7: "Other clinic",8: "Other hospital"})
df['date_crf'] = pd.to_datetime(df['date_crf'], errors='coerce', format='%Y-%m-%d')
df[['hiv_rdt','malaria_rdt','hepb_rdt','hepc_rdt','syphilis_rdt']]=df[['hiv_rdt','malaria_rdt','hepb_rdt','hepc_rdt','syphilis_rdt']].replace({1:'Positive',2:'Negative'})
df=df.dropna(subset=['siteregion_crf'])
df=df.dropna(subset=['date_crf'])
# Filter DataFrame based on current date and time
df = df[(df['date_crf']<=pd.to_datetime(datetime.now()))]
df=df.replace({'Ondo':'OWO','Lagos':'IKORODU','Ebonyi':'ABAKALIKI','Edo':'IRRUA'})
#df['Date of visit'] =pd.to_datetime(df['Date of visit'] ).dt.strftime('%Y-%m-%d')
col1, col2=st.sidebar.columns((2))
#Startdate=pd.to_datetime(df['date_crf']).min()
df['siteregion_crf']=df['siteregion_crf'].replace({1:'IKORODU',4:'ABAKALIKI',2:'OWO',3:'IRRUA'})
for col, replacement in replacements.items():
    if col in df.columns:
        df[col] = df[col].replace(replacement)
Enddate=pd.to_datetime(df['date_crf']).max()
Startdate=  pd.to_datetime(Enddate - timedelta(weeks=2))
with col1:
      date1=pd.to_datetime(st.date_input("Start Date", Startdate))
with col2:
      date2=pd.to_datetime(st.date_input("End Date", Enddate))

df=df[(df['date_crf']>=date1)&(df['date_crf']<=date2)].copy()

st.sidebar.header("Choose your filter : ")
##Create for State

state=st.sidebar.multiselect("Choose a State",   df["siteregion_crf"].unique())
if not state:
    df1=df.copy()
else:
   df1=df[df['siteregion_crf'].isin(state)]
   df=df[df['siteregion_crf'].isin(state)]
 
 
dfsample=df[['siteregion_crf','bloodsample','urinesample',
'nasosample',
'salivasample',
'oralsample']]
samplecol=['siteregion_crf','bloodsample','urinesample',
'nasosample',
'salivasample',
'oralsample']
testcol=['siteregion_crf','hiv_rdt','malaria_rdt','hepb_rdt','hepc_rdt','syphilis_rdt']
dftest=df[['siteregion_crf','hiv_rdt','malaria_rdt','hepb_rdt','hepc_rdt','syphilis_rdt']]
dfpcr=df[['siteregion_crf','yellowfever_pcr','lassa_pcr','ebola_pcr','marburg_pcr','westnile_pcr','zika_pcr','cchf_pcr','riftvalley_pcr','dengue_pcr','ony_pcr','covid_pcr','mpox_pcr']]
pcrcol=['siteregion_crf','yellowfever_pcr','lassa_pcr','ebola_pcr','marburg_pcr','westnile_pcr','zika_pcr','cchf_pcr','riftvalley_pcr','dengue_pcr','ony_pcr','covid_pcr','mpox_pcr']
def weeksago(df):
  two_weeks_ago =  pd.to_datetime(Startdate - timedelta(weeks=2))
  sliced_df = sliced_df =df[(df['date_crf']<=date1)&(df['date_crf']>=two_weeks_ago)].copy()
  
  return sliced_df 
def samplecollected(df):
     dfsample1=dfsample.drop(columns=['siteregion_crf'])
     missingfomrs = dfsample1[dfsample1.isna().all(axis=1)]
     return missingfomrs
     
def sampletest(df):
     dftest1=dftest.drop(columns=['siteregion_crf'])
     missingfomrs = dftest1[dftest1.isna().all(axis=1)]
     return missingfomrs
def samplepcr(df):
     dfpcr1=dfpcr.drop(columns=['siteregion_crf'])
     missingfomrs = dfpcr1[dfpcr1.isna().all(axis=1)]
     return missingfomrs          

# Calculate the date two weeks ago
two_weeks_ago = pd.to_datetime(date1 - timedelta(weeks=2))
dfsample1=samplecollected(df)
dftest1=sampletest(df)
dfpcr1=samplepcr(df)

sliced_df =df[(df['date_crf']>=date1)&(df['date_crf']<=two_weeks_ago)].copy()
data1 = {' ':['Patient enrolled','Sample collected','RDTs run','PCRs run'],
        'Last two weeks': [len(df),"{} ({:.2f}%)".format(len(df)*5,(((len(df)*5-dfsample1.isnull().sum().sum())/(len(df)*5))*100)),"{} ({:.2f}%)".format(len(df)*5,(((len(df)*5-dftest1.isnull().sum().sum())/(len(df)*5))*100)),"{} ({:.2f}%)".format(len(df)*12,(((len(df)*12-dfpcr1.isnull().sum().sum())/(len(df)*12))*100))]
    
      #  'previous two weeks': [len(weeksago(df)),"{} ({:.2f}%)".format(len(weeksago(df))*5,(((len(weeksago(df))-len(samplecollected(weeksago(df))))/len(weeksago(df)))*100))]
        }
#"{} {.2f%} ".format(len(df),((len(dfsample)*5-samplecollected(df)*5/len(df)*5)*100)),((len(weeksago(df))*5-samplecollected(weeksago(dfsample))*5/len(weeksago(dfsample))*5)*100)

    # Create DataFrame
data11 = {' ':['Patient enrolled','Sample collected','RDTs run','PCRs run'],
        'Last two weeks': [len(df),"{} ({:.2f}%)".format(len(df)*5,((len(df)-len(dfsample1))/len(df))*100),"{} ({:.2f}%)".format(len(df)*5,((len(df)-len(dftest1))/(len(df)))*100),"{} ({:.2f}%)".format(len(df)*12,((len(dfpcr1)-len(dfpcr1))/len(df))*100)]
    
      #  'previous two weeks': [len(weeksago(df)),"{} ({:.2f}%)".format(len(weeksago(df))*5,(((len(weeksago(df))-len(samplecollected(weeksago(df))))/len(weeksago(df)))*100))]
        }
dataframe1 = pd.DataFrame(data1)

    # Display the DataFrame as a table
#st.dataframe(dataframe1) 

data2 = {' ':['HIV','Malaria','Hepatitis B','Hepatitiis C','Syphilis'],
        'Last two weeks': ["{} ({:.2f}%)".format((df['hiv_rdt'] == 'Positive').sum(),(df['hiv_rdt'] == 'Positive').sum()*100/len(df)),"{} ({:.2f}%)".format((df['malaria_rdt'] == 'Positive').sum(),(df['malaria_rdt'] == 'Positive').sum()*100/len(df)),"{} ({:.2f}%)".format((df['hepb_rdt'] == 'Positive').sum(),(df['hepb_rdt'] == 'Positive').sum()*100/len(df)),"{} ({:.2f}%)".format((df['hepc_rdt'] == 'Positive').sum(),(df['hepc_rdt'] == 'Positive').sum()*100/len(df)),"{} ({:.2f}%)".format((df['syphilis_rdt'] == 'Positive').sum(),(df['syphilis_rdt'] == 'Positive').sum()*100/len(df))]
    
      #  'previous two weeks': [len(weeksago(df)),"{} ({:.2f}%)".format(len(weeksago(df))*5,(((len(weeksago(df))-len(samplecollected(weeksago(df))))/len(weeksago(df)))*100))]
        }
#"{} {.2f%} ".format(len(df),((len(dfsample)*5-samplecollected(df)*5/len(df)*5)*100)),((len(weeksago(df))*5-samplecollected(weeksago(dfsample))*5/len(weeksago(dfsample))*5)*100)

    # Create DataFrame
dataframe2 = pd.DataFrame(data2)

    # Display the DataFrame as a table
#st.dataframe(dataframe2)
st.write(df.head())    
st.write("## Overall summary:summary")
col1, col2=st.columns((2))

with col1:
    st.dataframe(dataframe1) 
 
with col2:
    st.dataframe(dataframe2) 
  
for state in df['siteregion_crf'].unique():
   dfs=df[df['siteregion_crf']==state]  
   #dfsample=dfsample[dfsample['siteregion_crf']==state]
   #dftest=dftest[dftest['siteregion_crf']==state]
   #st.write(dfsample)
   dfsample=dfs[samplecol]
   dfsample=dfs[samplecol]
   dftest=dfs[testcol]
   dfpcr=dfs[pcrcol]
   dftest=dftest.drop(columns=['siteregion_crf'])
   missingformt = dftest[dftest.isna().all(axis=1)]
   dfpcr=dfpcr.drop(columns=['siteregion_crf'])
   missingformp = dfpcr[dfpcr.isna().all(axis=1)]
   dfsample=dfsample.drop(columns=['siteregion_crf'])
   missingforms = dfsample[dfsample.isna().all(axis=1)]

   data = {' ':['Patient enrolled','Missing sample collection forms',' Missing RDTs forms','Missing PCRs forms'],
        'Last two weeks': [len(dfs),"{} ({:.2f}%)".format( missingforms.isna().sum().sum(),(( missingforms.isna().sum().sum()/(len(dfs)*5))*100)),"{} ({:.2f}%)".format( missingformt.isna().sum().sum(),(( missingformt.isna().sum().sum()/(len(dfs)*5))*100)),"{} ({:.2f}%)".format( missingformp.isna().sum().sum(),(( missingformp.isna().sum().sum()/(len(dfs)*12))*100))]}
   dataframe = pd.DataFrame(data) 
   st.write(f"## {state}")
   st.dataframe(dataframe)    

dff= df.groupby(['date_crf', 'siteregion_crf']).size().reset_index(name='Count')
#dff['Region of site']=dff['Region of site'].replace({'Ikorodu':'IKORODU','Abakaliki ':'Abakaliki','Abakaliki ':'Abakalik','Ebonyi ':'Ebonyi'})

#dff= dff[(dff['LGA']=="IKORODU") | (dff['LGA'] == 'OWO') |(dff['LGA'] == 'Abakaliki')|(dff['LGA'] == 'Ebonyi')]
#dff= dff[(dff['Region of site']=='IKORODU') | (dff['Region of site'] == 'Abakaliki')]
dff['date_crf'] =pd.to_datetime(dff['date_crf'] ).dt.strftime('%Y-%m-%d').astype("datetime64")
for val1 in dff['date_crf'].unique():
    for val2 in dff['siteregion_crf'].unique():
     new_row = {'date_crf': val1, 'siteregion_crf': val2,'Count':0}
     dff = dff.append(new_row, ignore_index=True)
    #dff['date_crf'] = pd.to_datetime(dff['date_crf'])#now
dff=dff.groupby(['date_crf','siteregion_crf']).sum().reset_index()[['Count','date_crf','siteregion_crf']]
dfff=pd.merge(dff,dff.groupby(['date_crf']).sum().reset_index(),on="date_crf")
# Additional plot settings
dfff['Total']='Total'
fig,ax = plt.subplots(figsize=(12, 9))
sns.lineplot( x="date_crf", y="Count_x", data=dfff , hue='siteregion_crf',palette='Set1').set(title=' ', xlabel='Visit date', ylabel='siteregion_crf')
sns.set_theme(style='white', font_scale=3)
ax.legend(loc='upper center', fancybox=True, shadow=True, ncol=5, fontsize=12)

# Calculate duration
start_date = dff['date_crf'].iloc[0]
end_date = dff['date_crf'].iloc[-1]
duration = end_date - start_date

# Determine tick frequency
if duration <= pd.Timedelta(days=90):  # Less than or equal to 2 months
    tick_freq = '2D'  # Daily
else:
    tick_freq = '15D'  # Monthly

# Generate ticks
ticks = pd.date_range(start=start_date, end=end_date, freq=tick_freq)
labels = [date.strftime('%Y-%m-%d') for date in ticks]

# Set ticks and labels
ax.set_xticks(ticks)
ax.set_xticklabels(labels, rotation=45 if tick_freq == 'M' else 90)

# Customize x-axis and y-axis
ax.tick_params(axis='x', labelsize=12)
ax.set_xlabel('Visit date', fontsize=12)  # Adjust as needed
ax.set_ylabel('Frequency', fontsize=12) 
ax.set_xlabel('Visit date')
ax.set_ylabel('Frequency')

# Adjust layout and display
fig.tight_layout()
st.pyplot(fig)
# Save the plot to a file-like object
buf1 = io.BytesIO()
fig.savefig(buf1, format='png')
buf1.seek(0)

# Create a download button for the plot
st.download_button(label="Download Plot",
    data=buf1,
    file_name="plot_date.png",
    #mime="image/png"
)

dff= df.groupby(['hiv_rdt', 'siteregion_crf']).size().reset_index(name='Count')

dfff=pd.merge(dff,dff.groupby(['siteregion_crf']).sum().reset_index(),on="siteregion_crf")


dfc=df[(df['participantid_crf']!=np.nan)&(df['participantid_rdt']!=np.nan)&(df['participantid_rdt']!=df['participantid_crf'])&(df['age_rdt']!=df['age_crf'])|(df['sex_rdt']!=df['sex_crf'])]
dfc=dfc.dropna(subset=['participantid_crf','participantid_rdt'])
st.write(f"## Mismatch check between forms: number of mismach is {len(dfc)} ")
st.write(dfc[['participantid_crf','participantid_rdt','age_rdt','age_crf','sex_crf','sex_rdt' ,'siteregion_crf']])     
        
select_out=st.multiselect('Please select a to check outliers:',df.select_dtypes(include='number').columns)
# Function to detect outliers using z-score
def detect_outliers_zscore(data):
    # Calculate z-score for each value in the dataframe
    z_scores = np.abs(zscore(data))
    # Threshold for considering a value as an outlier (e.g., z-score > 3)
    threshold = 3
    # Create a boolean mask indicating outliers
    outlier_mask = (z_scores > threshold)
    return outlier_mask

# Main app
if select_out:
    data =df[select_out].dropna()



    # Checkbox to show outliers
    show_outliers = st.checkbox('Show Outliers')

    # Detect outliers using z-score
    outliers = detect_outliers_zscore(data)
    st.write(outliers)
    if len(outliers)!=0:
        # Show outliers in the dataframe
        st.write('Outliers:', data[outliers.any(axis=1)])
    else:
        st.write('No outliers detected.')
        
           


# Sample categorical column selection
selected_columns = st.multiselect('Please select categorical/symptom columns to make line plots:', df.select_dtypes(include='object').columns)

if selected_columns:
    # Create rows for side-by-side plots
    num_columns = 2  # Number of plots per row
    num_rows = (len(selected_columns) + num_columns - 1) // num_columns  # Calculate number of rows needed

    def plot_percentage_line_plot(categorical_column):
        # Filter out unwanted values
        dff = df[(df[categorical_column] != 3) & (df[categorical_column] != 4)]
        
        # Group by date and selected column, calculate count
        dff = dff.groupby(['date_crf', categorical_column]).size().reset_index(name='Count')
        
        # Convert date column to datetime
        dff['date_crf'] = pd.to_datetime(dff['date_crf']).dt.strftime('%Y-%m-%d').astype("datetime64")

        # Ensure all combinations of date and category are present
        for val1 in dff['date_crf'].unique():
            for val2 in dff[categorical_column].unique():
                new_row = {'date_crf': val1, categorical_column: val2, 'Count': 0}
                dff = dff.append(new_row, ignore_index=True)

        # Group by date and category, summing counts
        dff = dff.groupby(['date_crf', categorical_column]).sum().reset_index()[['Count', 'date_crf', categorical_column]]

        # Calculate total count per date
        dff['Total'] = dff.groupby('date_crf')['Count'].transform('sum')

        # Calculate the percentage
        dff['Percentage'] = dff['Count'] / dff['Total'] * 100

        # Plotting the percentage over time
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x="date_crf", y="Percentage", data=dff, hue=categorical_column, palette='Set1')
        
        # Set title and labels
        ax.set_title(f'Percentage Over Time for {categorical_column}', fontsize=16)  # Main title with column name
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Percentage', fontsize=12)
        
        sns.set_theme(style='white', font_scale=1.5)
        ax.legend(loc='upper center', fancybox=True, shadow=True, ncol=5, fontsize=10)

        # Calculate duration
        start_date = dff['date_crf'].iloc[0]
        end_date = dff['date_crf'].iloc[-1]
        duration = end_date - start_date

        # Determine tick frequency
        tick_freq = '2D' if duration <= pd.Timedelta(days=90) else '15D'  # Daily or Monthly

        # Generate ticks
        ticks = pd.date_range(start=start_date, end=end_date, freq=tick_freq)
        labels = [date.strftime('%Y-%m-%d') for date in ticks]

        # Set ticks and labels
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45 if tick_freq == 'M' else 90)

        # Customize x-axis and y-axis
        ax.tick_params(axis='x', labelsize=10)
        fig.tight_layout()  # Adjust layout
        return fig  # Return the figure

    # Loop through selected columns and plot in rows of two
    for i in range(num_rows):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            index = i * num_columns + j
            if index < len(selected_columns):
                categorical_column = selected_columns[index]
                fig = plot_percentage_line_plot(categorical_column)
                cols[j].pyplot(fig)  # Display the plot in the respective column

                # Save the plot to a file-like object
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)

                # Create a download button for the plot
                cols[j].download_button(label=f"Download Plot for {categorical_column}",
                                        data=buf,
                                        file_name=f"plot_{categorical_column}.png")

select_catcol=st.multiselect('Please select categorical column to make  a bar plot:',df.select_dtypes(include='object').columns)
if select_catcol:
    st.write("Selected categorical column is : ", select_catcol[0])
 
   # selected_columns = st.multiselect("select column", select_col)
    s = df[select_catcol[0]].str.strip().value_counts()
    count_df = pd.DataFrame({f'{select_catcol[0]}': s.index, 'Count': s.values})
    st.write(count_df)
    trace = go.Bar(x=s.index,y=s.values,showlegend = False, text=s.values)
    layout = go.Layout(title = f"Bar plot for {select_catcol[0]}")
    data = [trace]
    fig = go.Figure(data=data,layout=layout)
    st.plotly_chart(fig)
    

else:
    st.info('Please select a categorical column using the dropdown above.')



# Sample categorical column selection
selected_columns = st.multiselect('Please select categorical/symptom columns to make pie charts:', df.select_dtypes(include='object').columns)


if selected_columns:
    # Create rows for side-by-side plots
    num_columns = 2  # Number of plots per row
    num_rows = (len(selected_columns) + num_columns - 1) // num_columns  # Calculate number of rows needed

    def plot_percentage_pie_chart(categorical_column):
        # Filter out unwanted values
        dff = df[(df[categorical_column] != 3) & (df[categorical_column] != 4)]
        
        # Calculate counts for each category
        category_counts = dff[categorical_column].value_counts()
        
        # Create a pie chart with reduced figure size
        fig, ax = plt.subplots(figsize=(5, 4))  # Reduced size (width, height)

        # Create a pie chart without labels and include percentages
        wedges, _ = ax.pie(
            category_counts,
            labels=None,  # Remove labels from the pie chart
            #autopct='%1.1f%%',  # Format for percentage display
            startangle=90,  # Start angle for the pie chart
            colors=plt.cm.Set1.colors  # Color map for pie sections
        )

        # Set font size for percentage text
        #for text in autotexts:
           # text.set_fontsize(12)  # Adjust font size for percentage text

        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # Set title with reduced font size
        ax.set_title(f'Distribution of {categorical_column}', fontsize=10)  # Title with adjusted fontsize

        # Create custom legend labels with count and percentage
        legend_labels = [f"{category}: {count} ({count / dff.shape[0] * 100:.2f}%)" for category, count in category_counts.items()]

        # Add legend with smaller text
        ax.legend(
            wedges, 
            legend_labels,  # Use the custom legend labels
            loc="upper right", 
            bbox_to_anchor=(1.5, 1), 
            fontsize=8  # Font size for legend
        )

        plt.tight_layout()  # Adjust layout for better spacing
        return fig  # Return the figure

    # Loop through selected columns and plot in rows of two
    for i in range(num_rows):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            index = i * num_columns + j
            if index < len(selected_columns):
                categorical_column = selected_columns[index]
                fig = plot_percentage_pie_chart(categorical_column)
                cols[j].pyplot(fig)  # Display the plot in the respective column

                # Save the plot to a file-like object
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)

                # Create a download button for the plot
                cols[j].download_button(
                    label=f"Download Pie Chart for {categorical_column}",
                    data=buf,
                    file_name=f"pie_chart_{categorical_column}.png"
                )               
select_numcol=st.multiselect('Please select numerical column to make  a histogram plot:',df.select_dtypes(include='number').columns)




#select_numcol=st.multiselect('Select numerical column:',dfs.select_dtypes(include='number').columns)
#select_numcol=st.multiselect('Select numerical column:',dfs.columns)

if select_numcol:
   st.write("Selected numerical column is : ", select_numcol[0])
   fig, ax = plt.subplots()
   q1 = df[select_numcol[0]].quantile(0.25)
   q3 = df[select_numcol[0]].quantile(0.75)
   iqr = q3 - q1
   bin_width = (2 * iqr) / (len(dfs[select_numcol[0]]) ** (1 / 3))
   bin_count = int(np.ceil((dfs[select_numcol[0]].max() - dfs[select_numcol[0]].min()) / bin_width))
   selected_value = st.slider('Select number of bins:', min_value=0, max_value=100, value=bin_count)

# Display the selected value
   st.write(f'Select number of bins: {selected_value}')
   ax.hist(dfs[select_numcol[0]], bins=selected_value, edgecolor='black')
   st.pyplot(fig)
   buf2 = io.BytesIO()
   fig.savefig(buf2, format='png')
   buf2.seek(0)

# Create a download button for the plot
   st.download_button(
   label="Download Plot",
    data=buf2,
    file_name="plot_hist.png",
    mime="image/png"
)
else:
    st.info('Please select a numerical column using the dropdown above.')


# Select categorical and numerical columns
select_catcol = st.multiselect('Please select categorical column to make a box plot:', df.select_dtypes(include='object').columns)
select_numcol = st.multiselect('Please select numerical column to make a box plot:', df.select_dtypes(include='number').columns)

def t_test(sample1, sample2):
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)
    return t_statistic, p_value

def mann_whitney_test(sample1, sample2):
    u_statistic, p_value = stats.mannwhitneyu(sample1, sample2)
    return u_statistic, p_value

if select_catcol and select_numcol:
    # Use selectbox for choosing categories
    categories = df[select_catcol[0]].unique()
    cat1 = st.selectbox("Choose the first category", categories)
    cat2 = st.selectbox("Choose the second category", categories)

    if cat1 == cat2:
        st.warning("Please select two different categories.")
    else:
        df1 = df[df[select_catcol[0]].isin([cat1, cat2])]
        df1[select_numcol[0]] = df1[select_numcol[0]].fillna(df1[select_numcol[0]].mean())
        
        sample1 = df1.loc[df1[select_catcol[0]] == cat1][select_numcol[0]]
        sample2 = df1.loc[df1[select_catcol[0]] == cat2][select_numcol[0]]

        st.write("Selected categorical column is: ", select_catcol[0], " and Selected numerical column is: ", select_numcol[0])
        
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=select_catcol[0], y=select_numcol[0], data=df1, palette="Set1", width=0.2)
        ax.set_title(' ')
        ax.set_xlabel(f'{select_catcol[0]}', fontsize=12)
        ax.set_ylabel(f'{select_numcol[0]}', fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        fig.tight_layout()

        # Display the plot in Streamlit app
        st.pyplot(fig)

        buf3 = io.BytesIO()
        fig.savefig(buf3, format='png')
        buf3.seek(0)

        # Create a download button for the plot
        st.download_button(
            label="Download Plot",
            data=buf3,
            file_name="plot_box.png",
            mime="image/png"
        )

        # Perform t-test
        t_statistic, t_p_value = t_test(sample1, sample2)

        # Perform Mann-Whitney U test
        u_statistic, u_p_value = mann_whitney_test(sample1, sample2)

        # Display results
        st.write("### Statistical test")
        st.write("#### Independent Samples t-test")
        st.write(f"T-Statistic: {t_statistic}")
        st.write(f"P-Value: {t_p_value}")

        st.write("#### Mann-Whitney U Test")
        st.write(f"U-Statistic: {u_statistic}")
        st.write(f"P-Value: {u_p_value}")


# Function to perform chi-square test
def chi_square_test(data1,data2):
    data=pd.crosstab(data1, data2)
    chi2, p, dof, expected = chi2_contingency(data)
    return chi2, p

# Function to perform Fisher's exact test
def fishers_exact_test(data):
    oddsratio, p = fisher_exact(data)
    return oddsratio, p

# Function to calculate confidence intervals for odds ratio
def calculate_confidence_interval(odds_ratio, a, b, c, d, alpha=0.05):
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = -1 * np.percentile(np.random.normal(size=100000), alpha * 100)
    ci_lower = np.exp(np.log(odds_ratio) - z * se_log_or)
    ci_upper = np.exp(np.log(odds_ratio) + z * se_log_or)
    return ci_lower, ci_upper

# Multiselect for first categorical column
select_col11 = st.multiselect(
    'Please select a first categorical column for statistical analysis', 
    df.select_dtypes(include='object').columns
)

# Multiselect for second categorical column
select_col22 = st.multiselect(
    'Please select a second categorical column for statistical analysis', 
    df.select_dtypes(include='object').columns
)

# Main app
if select_col11 and select_col22:
    cat11 = st.multiselect(
        "Choose two categories for the first categorical variable", 
        df[select_col11[0]].unique()
    )
    cat22 = st.multiselect(
        "Choose two categories for the second categorical variable", 
        df[select_col22[0]].unique()
    )
    
    df1 = df.copy()
    if cat11:
        df1 = df1[df1[select_col11[0]].isin(cat11)]
    if cat22:
        df1 = df1[df1[select_col22[0]].isin(cat22)]
    
    selected_data = df1[[select_col11[0], select_col22[0]]].dropna()
    # Perform chi-square test
    st.subheader('Chi-Square Test')
    chi2, p = chi_square_test(selected_data[select_col11[0]], selected_data[select_col22[0]])
    st.write('Chi-Square Statistic:', chi2)
    st.write('P-value:', p)
    
    # Check if we have exactly two categories for both variables
    unique_cat11 = selected_data[select_col11[0]].nunique()
    unique_cat22 = selected_data[select_col22[0]].nunique()

    if unique_cat11 == 2 and unique_cat22 == 2:
        # Create crosstab for Fisher's exact test
        cross_tab = pd.crosstab(index=selected_data[select_col22[0]], columns=selected_data[select_col11[0]])

        # Perform Fisher's exact test
        st.subheader('Fisher\'s Exact Test')
        odds_ratio, fisher_p = fisher_exact(cross_tab)
        st.write('Odds Ratio:', odds_ratio)
        st.write('Fisher\'s Exact Test P-value:', fisher_p)
    else:
        st.warning('Fisher\'s exact test can only be performed with exactly two categories for both variables.')
    
    # Create crosstab and proportion table
    cross_tab = pd.crosstab(index=selected_data[select_col22[0]], columns=selected_data[select_col11[0]])
    cross_tab_prop = pd.crosstab(index=selected_data[select_col22[0]], columns=selected_data[select_col11[0]], normalize="index")
    
    # Plotting
    st.subheader(f'Crosstab plot {select_col11[0]} vs {select_col22[0]}')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.tick_params(axis='x', labelsize=8, pad=3)  # Reduce label size and pad
    cross_tab_prop.plot(kind='bar', stacked=True, ax=ax, rot=0, width=0.3)
    ax.legend(fontsize=10) 
    # Add text annotations
    for n, x in enumerate(cross_tab.index.values):
        for (proportion, count, y_loc) in zip(cross_tab_prop.loc[x], cross_tab.loc[x], cross_tab_prop.loc[x].cumsum()):
            ax.text(x=n - 0.17, y=(y_loc - proportion) + (proportion / 2),
                    s=f'{count} ({np.round(proportion*100, 2)}%)',
                    color="black", fontsize=8, fontweight="bold")

    st.pyplot(fig)
    # Check if we have exactly two categories for both variables
    unique_cat11 = selected_data[select_col11[0]].nunique()
    unique_cat22 = selected_data[select_col22[0]].nunique()

    if unique_cat11 == 2 and unique_cat22 == 2:
        # Create crosstab for Fisher's exact test
        cross_tab = pd.crosstab(index=selected_data[select_col22[0]], columns=selected_data[select_col11[0]])

        # Perform Fisher's exact test for 2x2 tables
        st.subheader('Fisher\'s Exact Test (2x2 Table)')
        odds_ratio, fisher_p = fisher_exact(cross_tab)
        st.write('Odds Ratio:', odds_ratio)
        st.write('Fisher\'s Exact Test P-value:', fisher_p)
    else:
        st.warning('Fisher\'s exact test can only be performed with exactly two categories for both variables.')

    # Check if we can proceed with the reference group selection for multiple categories
    if not selected_data.empty:
        # Create a dropdown menu to select the reference group
        cross_tab = pd.crosstab(index=selected_data[select_col22[0]], columns=selected_data[select_col11[0]])
        ref_group = st.selectbox('Select the reference group for Fisher\'s Exact Test', cross_tab.index)

        # Check if Fisher's exact test can be performed (multiple categories)
        if cross_tab.shape[0] > 2 or cross_tab.shape[1] > 2:
            st.subheader('Fisher\'s Exact Test for Multiple Categories')

            # Get the reference group data
            ref_group_data = cross_tab.loc[ref_group].values

            # Initialize results dictionary
            results = {}

            # Loop over each group and perform Fisher's exact test
            for group in cross_tab.index:
                if group != ref_group:
                    group_data = cross_tab.loc[group].values
                    table = [[group_data[0], group_data[1]], [ref_group_data[0], ref_group_data[1]]]
                    
                    # Perform Fisher's Exact Test
                    oddsratio, p_value = fisher_exact(table)
                    ci_lower, ci_upper = calculate_confidence_interval(
                        oddsratio,
                        group_data[0], group_data[1],
                        ref_group_data[0], ref_group_data[1]
                    )
                    
                    results[group] = {
                        'odds_ratio': oddsratio,
                        'p_value': p_value,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    }

            # Convert results to a DataFrame
            results_df = pd.DataFrame(results).T

            # Adjust p-values for multiple testing
            p_values = results_df['p_value'].values
            _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
            results_df['p_adj'] = p_adj

            # Display results
            st.dataframe(results_df)

        else:
            st.warning('Fisher\'s exact test for multiple categories can only be performed with more than two categories.')
    else:
        st.warning('The crosstab is empty. Please select valid categories.')
    
st.title("Feature Selection with Streamlit")
df = df.dropna(axis=1, how='all')
# Choose dependent variable
default_target = df.columns[0]  # First column as default target variable
default_independent = [col for col in df.columns if col != default_target]
# Filter out categorical columns with more than 10 unique categories
nokit = df.loc[:, ~df.columns.str.contains('kit', case=False)]
filtered_columns = [col for col in nokit.columns if nokit[col].nunique() <= 7]

# Filter categorical columns with 7 or fewer unique categories
categorical_columns = nokit.select_dtypes(include=['object', 'category']).columns
filtered_categorical_columns = [col for col in categorical_columns if nokit[col].nunique() <= 7]

# Keep all numerical columns
numerical_columns = nokit.select_dtypes(include=['number']).columns

# Combine filtered categorical columns and numerical columns
filtered_columns = list(filtered_categorical_columns) + list(numerical_columns)


target_variable = st.selectbox("Choose the dependent variable", options=filtered_columns,index=nokit.columns.get_loc(default_target))

# Choose independent variables
#independent_variables = st.multiselect("Choose independent variables", options=[col for col in df.columns if col != target_variable],default=default_independent)
# Default selection
#default_target = df.columns[5]  # First column as default target variable
#default_independent = [col for col in df.columns if col != default_target][6:16]  # All other columns as default independent variables



# Ensure the default values are valid options
available_independent_vars = [col for col in nokit.columns if col != target_variable]
default_independent = [var for var in default_independent if var in available_independent_vars]

# Choose independent variables
independent_variables = st.multiselect("Choose independent variables", options=available_independent_vars, default=default_independent)
print(df[target_variable],'printed')
df.dropna(subset=[target_variable],axis=0,inplace=True)
if pd.api.types.is_object_dtype(df[target_variable]):
   cat=st.multiselect("Choose two categories",options=df[target_variable].unique(), 
    default=df[target_variable].unique())
   df=df[df[target_variable].isin(cat)]
   if not cat:
       df=df.copy()
else:
     df=df
X = df[independent_variables]
y = df[target_variable]
st.write(X.values.shape,y.values.shape)
# Check if y is categorical or numerical
if y.dtype == 'object':
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    is_classification = True
else:
    y=y
    is_classification = False
# Identify continuous and categorical features
continuous_features = list(X.select_dtypes(include=['float64']).columns)
categorical_features = list(X.select_dtypes(include=['object']).columns)

# Handle empty continuous features
if continuous_features:
    Xc = X[continuous_features]
    Xc = Xc.dropna(axis=1, how='all')
    Xc = Xc.fillna(Xc.mean())
    scaler = StandardScaler()
    Xc_normalized = pd.DataFrame(scaler.fit_transform(Xc), columns=Xc.columns, index=Xc.index)
else:
    Xc_normalized = pd.DataFrame(index=X.index)

# Handle empty categorical features
if categorical_features:
    Xn = X[categorical_features].fillna(X[categorical_features].mode().iloc[0])
    Xn_dummies = pd.get_dummies(Xn, drop_first=True, prefix=categorical_features, prefix_sep='_')
    Xn_dummies = Xn_dummies.reindex(index=X.index)  # Ensure index alignment
else:
    Xn_dummies = pd.DataFrame(index=X.index)

# Merge normalized continuous features with dummy variables
X_transformed = pd.concat([Xc_normalized, Xn_dummies], axis=1, join='inner')
st.write(X_transformed.values.shape,Xc_normalized.values.shape, Xn_dummies.values.shape,y.shape)
# Ensure that y has the same index as X_transformed
#y = y.loc[X_transformed.index]

# Drop-down menu for feature selection method
method = st.selectbox("Choose feature selection method", ["SelectKBest", "RFE"])

# Default number of features to keep
# Default number of features to keep

#n_features = st.slider("Select number of features to keep", min_value=1, max_value=X_transformed.shape[1], value=default_k)
default_k = X_transformed.shape[1] if X_transformed.shape[1] > 0 else 1
k = st.slider("Select number of features to keep", min_value=1, max_value=X_transformed.shape[1], value=default_k)
# Apply feature selection based on user choice
if method == "SelectKBest":
    if is_classification:
        score_func = f_classif
    else:
        score_func = f_regression
    
    #k = st.slider("Select number of features to keep", min_value=1, max_value=X_transformed.shape[1], value=default_k)
    selector = SelectKBest(score_func, k=k)
    X_selected = selector.fit_transform(X_transformed, y)
    selected_features = X_transformed.columns[selector.get_support()]
    
    # Plot feature importances for SelectKBest
    scores = selector.scores_
    feature_scores = pd.DataFrame({'Feature': X_transformed.columns, 'Score': scores})
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)
    top_k_scores =  feature_scores.head(k)
    st.write(f"Selected features: {', '.join(selected_features)}")
    
    # Plotting
    plt.figure(figsize=(15, 15))
    palette = sns.color_palette("viridis", len(top_k_scores)) 
    sns.barplot(x='Score', y='Feature', data=top_k_scores,palette=palette)
    ax.tick_params(axis='x', labelsize=8)  # Reduce x-axis tick label size
    ax.tick_params(axis='y', labelsize=8)  # Reduce y-axis tick label size
    plt.title('Feature Scores from SelectKBest')
    st.pyplot(plt)
    
    fig.tight_layout()
# Display the plot in Streamlit app
    #st.pyplot(fig)
   #st.write(dfs.loc[dfs[select_catcol[0]]==dfs[select_catcol[0]].value_counts().index.tolist()[0]]['Sex'])
    buf6 = io.BytesIO()
    fig.savefig(buf6, format='png')
    buf6.seek(0)

# Create a download button for the plot
    st.download_button(
    label="Download Plot",
    data=buf6,
    file_name="plot_bestkafeats.png",
    mime="image/png"
)
    
elif method == "RFE":
    if is_classification:
        estimator = LogisticRegression(max_iter=1000)
    else:
        estimator = LinearRegression()

    #n_features = st.slider("Select number of features to keep", min_value=1, max_value=X_transformed.shape[1], value=default_k)
    if k > X_transformed.shape[1]:
        k= X_transformed.shape[1]  # Ensure n_features is not larger than the number of features
    selector = RFE(estimator, n_features_to_select=k)
    X_selected = selector.fit_transform(X_transformed, y)
    selected_features = X_transformed.columns[selector.support_]
    
    # Plot feature importances for RFE
    ranking = selector.ranking_
    
    feature_ranking = pd.DataFrame({'Feature': X_transformed.columns, 'Ranking': ranking})
    feature_ranking = feature_ranking.sort_values(by='Ranking')
    top_k_features = feature_ranking.head(k)
    st.write(f"Selected features: {', '.join(selected_features)}")
    
    # Plotting
    plt.figure(figsize=(15, 15))
    palette = sns.color_palette("viridis", len(top_k_features)) 
    sns.barplot(x='Ranking', y='Feature', data=top_k_features,palette=palette)
    ax.tick_params(axis='x', labelsize=8)  # Reduce x-axis tick label size
    ax.tick_params(axis='y', labelsize=8)  # Reduce y-axis tick label size
    plt.title('Feature Rankings from RFE')
    st.pyplot(plt)
    
    fig.tight_layout()
# Display the plot in Streamlit app
    #st.pyplot(fig)
   #st.write(dfs.loc[dfs[select_catcol[0]]==dfs[select_catcol[0]].value_counts().index.tolist()[0]]['Sex'])
    buf7 = io.BytesIO()
    fig.savefig(buf7, format='png')
    buf7.seek(0)

# Create a download button for the plot
    st.download_button(
    label="Download Plot",
    data=buf7,
    file_name="plot_RFEfeats.png",
    mime="image/png"
)

# Display transformed features
#st.write("Transformed Features")
st.dataframe(pd.DataFrame(X_selected, columns=selected_features))

def calc_vif(X_train):
    """Calculate Variance Inflation Factor (VIF) for each feature in X_train."""
    vif = pd.DataFrame()
    vif["variables"] = X_train.columns
    vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    return vif
    



# Calculate and display VIF
vif_df = calc_vif(X_transformed)


# Add drop-down for VIF threshold
vif_threshold = st.slider("Select VIF Threshold", min_value=1, max_value=10, value=2)
features_to_keep = vif_df[vif_df['VIF'] < vif_threshold]['variables'].tolist()
st.write(f"Features with VIF < {vif_threshold}: {features_to_keep}")

X_filtered = X_transformed[features_to_keep]

# Add dropdown for p-value threshold
p_value_threshold = st.slider("Select p-value Threshold", min_value=0.01, max_value=0.1, value=0.05)


        
        
# Fit OLS or logistic regression model
if st.button("Fit Model (with VIF filter)", key="fit_model_vif"):
    X_filtered_with_const = sm.add_constant(X_filtered)

    if is_classification:
        # Fit logistic regression model
        logit_model = sm.Logit(y, X_filtered_with_const)
        logit_result = logit_model.fit()
        st.write(logit_result.summary())
        
        # Filter features based on p-value
        importantv = pd.DataFrame(logit_result.summary2().tables[1])
        importantv['P>|z|']= sm.stats.multipletests(importantv['P>|z|'], method='fdr_by')[1]
        importantv = importantv.loc[(importantv['P>|z|'] < p_value_threshold)]
       
        # Create DataFrame for feature importance
        feat_importance = pd.DataFrame(importantv.index.tolist(), columns=["feature"])
        feat_importance["importance"] = abs(importantv['Coef.'].values)
        feat_importance = feat_importance.sort_values(by="importance", ascending=True)
        st.write(f"Features with p-value < {p_value_threshold}: {feat_importance['feature'].tolist()}")
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        ax = feat_importance.plot.barh(x="feature", y="importance", color='skyblue', legend=False)
        ax.set_xlabel('Absolute Coefficient')
        ax.set_title('Feature Importance')
        plt.gca().invert_yaxis()  # Invert y-axis to show most important features at the top
        plt.grid(True)
        st.pyplot(plt)
        
   
    
    else:
        # Fit OLS model
        ols_model = sm.OLS(y, X_filtered_with_const)
        ols_result = ols_model.fit()
        st.write(ols_result.summary())
        
        # Filter features based on p-value
        importantv = pd.DataFrame(ols_result.summary2().tables[1])
        importantv['P>|t|']= sm.stats.multipletests(importantv['P>|t|'], method='fdr_by')[1]
        importantv = importantv.loc[(importantv['P>|t|'] < p_value_threshold)]
        
        # Create DataFrame for feature importance
        feat_importance = pd.DataFrame(importantv.index.tolist(), columns=["feature"])
        feat_importance["importance"] = abs(importantv['Coef.'].values)
        feat_importance = feat_importance.sort_values(by="importance", ascending=True)
        st.write(f"Features with p-value < {p_value_threshold}: {feat_importance['feature'].tolist()}")
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        ax = feat_importance.plot.barh(x="feature", y="importance", color='skyblue', legend=False)
        ax.set_xlabel('Absolute Coefficient')
        ax.set_title('Feature Importance')
        plt.gca().invert_yaxis()  # Invert y-axis to show most important features at the top
        plt.grid(True)
        st.pyplot(plt)
        
       # st.write("Filtered Features based on p-value:")
       # st.dataframe(X_filtered_with_const[importantv.index])
