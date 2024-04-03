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
st.write('hello here')
url="https://docs.google.com/spreadsheets/d/1JIDg8pqO1Sn353-K8JkCCKZKEvqsytaMGiqBFT04f48/edit?usp=sharing"

conn = st.connection("gsheets", type=GSheetsConnection)
st.title("Data summary sentinel project")
st.markdown("---")

conn = st.connection("gsheets", type=GSheetsConnection)

st.markdown("---")
@st.cache_data
def load_data(datapath):
    dataset = conn.read(spreadsheet=datapath)
    return dataset

df0 = load_data(url)
def main():
    st.sidebar.title("Please  upload your own file  or Token")
    
    # File uploader widget
    uploaded_file =  st.sidebar.file_uploader("Upload a file", type=["txt", "csv", "xlsx"])

    # If a file is uploaded
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        st.write("File contents:")
        # Read and display file contents
        df = uploaded_file.read()
        st.write(df)
    
    # Text input widget for token
    token = st.sidebar.text_input("Input a token")

    # If a token is provided
    if token:
        st.write("Token provided:", token)
        data = {
    'token':token,
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
    'rawOrLabel': 'label',
    'rawOrLabelHeaders': 'raw',
    'exportCheckboxLabel': 'false',
    'exportSurveyFields': 'false',
    'exportDataAccessGroups': 'false',
    'returnFormat': 'csv'
}
        r = requests.post('https://redcap-acegid.org/api/',data=data)

        df = pd.read_csv(StringIO(r.text),  low_memory=False)
        st.write(df.head())
    else:
        df=df0     

    return df
  
  
df=main() 



       
#df = conn.read(spreadsheet=url)
df['date_crf'] = pd.to_datetime(df['date_crf'], errors='coerce', format='%Y-%m-%d')

# Filter DataFrame based on current date and time
df = df[(df['date_crf']<=pd.to_datetime(datetime.now()))]
df=df.replace({'Ondo':'OWO','Lagos':'IKORODU','Ebonyi':'ABAKALIKI','Edo':'IRRUA'})
#df['Date of visit'] =pd.to_datetime(df['Date of visit'] ).dt.strftime('%Y-%m-%d')
col1, col2=st.sidebar.columns((2))
#Startdate=pd.to_datetime(df['date_crf']).min()

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
 
 
dfsample=df[['siteregion_crf','bloodsample','urinesample',
'nasosample',
'salivasample',
'oralsample']]

dftest=df[['siteregion_crf','hiv_rdt','malaria_rdt','hepb_rdt','hepc_rdt','syphilis_rdt']]
dfpcr=df[['siteregion_crf','yellowfever_pcr','lassa_pcr','ebola_pcr','marburg_pcr','westnile_pcr','zika_pcr','cchf_pcr','riftvalley_pcr','dengue_pcr','ony_pcr','covid_pcr','mpox_pcr']]
def weeksago(df):
  two_weeks_ago =  pd.to_datetime(Startdate - timedelta(weeks=2))
  sliced_df = sliced_df =df[(df['date_crf']<=date1)&(df['date_crf']>=two_weeks_ago)].copy()
  
  return sliced_df 
def samplecollected(df):
     dftest=dfsample.drop(columns=['siteregion_crf'])
     missingfomrs = dftest[dftest.isna().all(axis=1)]
     return missingfomrs
     
def sampletest(df):
     dftest=dftest.drop(columns=['siteregion_crf'])
     missingfomrs = dftest[dftest.isna().all(axis=1)]
     return missingfomrs
def sampletest(df):
     dfpcr=dfpcr.drop(columns=['siteregion_crf'])
     missingfomrs = dfpcr[dfpcr.isna().all(axis=1)]
     return missingfomrs          

# Calculate the date two weeks ago
two_weeks_ago = pd.to_datetime(date1 - timedelta(weeks=2))

sliced_df =df[(df['date_crf']>=date1)&(df['date_crf']<=two_weeks_ago)].copy()
data1 = {' ':['Patient enrolled','Sample collected','RDTs run','PCRs run'],
        'Last two weeks': [len(df),"{} ({:.2f}%)".format(len(df)*5,(((len(df)*5-dfsample.isnull().sum().sum())/(len(df)*5))*100)),"{} ({:.2f}%)".format(len(df)*5,(((len(df)*5-dftest.isnull().sum().sum())/(len(df)*5))*100)),"{} ({:.2f}%)".format(len(df)*12,(((len(df)*12-dfpcr.isnull().sum().sum())/(len(df)*12))*100))]
    
      #  'previous two weeks': [len(weeksago(df)),"{} ({:.2f}%)".format(len(weeksago(df))*5,(((len(weeksago(df))-len(samplecollected(weeksago(df))))/len(weeksago(df)))*100))]
        }
#"{} {.2f%} ".format(len(df),((len(dfsample)*5-samplecollected(df)*5/len(df)*5)*100)),((len(weeksago(df))*5-samplecollected(weeksago(dfsample))*5/len(weeksago(dfsample))*5)*100)

    # Create DataFrame
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
st.write("## Overall summary")
col1, col2=st.columns((2))

with col1:
    st.dataframe(dataframe1) 
 
with col2:
    st.dataframe(dataframe2) 
  

for state in df['siteregion_crf'].unique():
   dfs=df[df['siteregion_crf']==state]  
   dfsample=dfsample[dfsample['siteregion_crf']==state]
   dftest=dftest[dftest['siteregion_crf']==state]

   dfpcr=dfpcr[dfpcr['siteregion_crf']==state]
   data = {' ':['Patient enrolled','Sample collected','RDTs run','PCRs run'],
        'Last two weeks': [len(dfs),"{} ({:.2f}%)".format(len(dfs)*5,(((len(dfs)*5-dfsample.isnull().sum().sum())/(len(dfs)*5))*100)),"{} ({:.2f}%)".format(len(dfs)*5,(((len(dfs)*5-dftest.isnull().sum().sum())/(len(dfs)*5))*100)),"{} ({:.2f}%)".format(len(dfs)*12,(((len(dfs)*12-dfpcr.isnull().sum().sum())/(len(dfs)*12))*100))]}
   dataframe = pd.DataFrame(data) 
   st.write(f"## {state}")
   st.dataframe(dataframe)
      
dff= df1.groupby(['date_crf', 'siteregion_crf']).size().reset_index(name='Count')
#dff['Region of site']=dff['Region of site'].replace({'Ikorodu':'IKORODU','Abakaliki ':'Abakaliki','Abakaliki ':'Abakalik','Ebonyi ':'Ebonyi'})

#dff= dff[(dff['LGA']=="IKORODU") | (dff['LGA'] == 'OWO') |(dff['LGA'] == 'Abakaliki')|(dff['LGA'] == 'Ebonyi')]
#dff= dff[(dff['Region of site']=='IKORODU') | (dff['Region of site'] == 'Abakaliki')]
dff['date_crf'] =pd.to_datetime(dff['date_crf'] ).dt.strftime('%Y-%m-%d')
dfff=pd.merge(dff,dff.groupby(['date_crf']).sum().reset_index(),on="date_crf")
dfff['Total']='Total'
fig,ax = plt.subplots(figsize=(30, 10))
sns.lineplot( x="date_crf", y="Count_x", data=dfff , hue='siteregion_crf',palette='Set1').set(title=' ', xlabel='Date', ylabel='siteregion_crf')
sns.lineplot( x="date_crf", y="Count_y", data=dfff,hue='Total',palette=['black'],).set(title=' ', xlabel='Date', ylabel='siteregion_crf')
#sns.set_theme(style='white', font_scale=3)
ax.legend(loc='upper center', #bbox_to_anchor=(0.4,0.0001),
          fancybox=True, shadow=True, ncol=5)
# Remove the frame (border) around the plot
#sns.gca().spines['top'].set_visible(False)
#sns.gca().spines['right'].set_visible(False)
#plt.gca().spines['bottom'].set_visible(False)
#plt.gca().spines['left'].set_visible(False)
#monthly_ticks = pd.date_range(start=dff['Date of visit (dd/mm/yyyy)'].iloc[0], end=dff['Date of visit (dd/mm/yyyy)'].iloc[-1],freq='d')  # Monthly intervals
#plt.xticks(ticks=monthly_ticks, labels=[date.strftime('%Y-%m-%d') for date in monthly_ticks], rotation=45)


ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Plot through Time with Custom X-axis Ticks')
plt.xticks(rotation=45)
#ax.tight_layout()
st.pyplot(fig)
    
df1['date_crf'] =pd.to_datetime(df1['date_crf'] ).dt.strftime('%Y-%m-%d')
df0=df1[df1['lassa_pcr']=='Positive']
dfg= df0.groupby(['date_crf', 'siteregion_crf', 'lassa_pcr']).size().reset_index(name='Positive lassa')
dfg['date_crf'] =pd.to_datetime(dfg['date_crf'] ).dt.strftime('%Y-%m-%d')
st.write(dfg)
state=dfg['siteregion_crf'].unique().tolist()
date=dfg['date_crf'].unique().tolist()
Date=st.selectbox('Which date',date, index=0)
State=st.multiselect('Which state',state,dfg['siteregion_crf'].unique())
dfstate=dfg[dfg['State of Residence'].isin(State)]
fig=px.bar(dfstate, x="siteregion_crf", y="Positive lassa", color="siteregion_crf",animation_frame="date_crf", animation_group="siteregion_crf")
#fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration']=30
#fig.layout.updatemenus[0].buttons[0].args[1]['transition'['duration']]=5
fig.update_layout(width=800)

st.write(fig)

# Multi-Select for Columns
selected_columns = st.multiselect('Select Columns', df.columns)

# Display Selected Columns
st.write('Selected Columns:', selected_columns)

df0=dff[dff['Lassa fever']=='Positive']
dfg= df0.groupby(['Date of visit', 'State of Residence', 'Lassa fever']).size().reset_index(name='Positive lassa')
fig = px.pie(dfg, names='State of Residence', values='Positive lassa', title='Pie Plot')
st.plotly_chart(fig)

# Filter DataFrame based on selected columns
filtered_df = df[selected_columns]

# Create Pie Plot
if not filtered_df.empty:
    fig = px.pie(filtered_df, names='Category', values='Value', title='Pie Plot')
    st.plotly_chart(fig)
else:
    st.warning('Select at least one column to display the pie plot.')
dfcat=dff.groupby(by = ['Lassa fever'] , as_index=False)['Age'].mean()
with col1:
  st.subheader("Category wise Age")
 # text =  ['Year:,.2f'].format(x) for x in dfcat,
  fig=px.bar(dfcat, x = "Lassa fever", y = "Age", template='seaborn')
  st.plotly_chart(fig, use_countainer_width=True, height=20)
with col2:
  st.subheader("State  wise Age")
  fig=px.pie(dff, values='Age', names="State of Residence", hole=0.5)
  fig.update_traces(text=dff['State of Residence'], textposition="outside")
  st.plotly_chart(fig, use_countainer_width=True, height=20)
dfs=df
# Display the selected columns
if selected_columns:
    #st.subheader('Selected Columns:')
    dfs=df[selected_columns]
else:
    st.info('Please select columns using the dropdown above.')
    dfs=df



select_catcol=st.multiselect('Please select categorical column to make  a bar plot:',dfs.select_dtypes(include='object').columns)

if select_catcol:
    st.write("Selected categorical column is : ", select_catcol[0])
 
   # selected_columns = st.multiselect("select column", select_col)
    s = dfs[select_catcol[0]].str.strip().value_counts()
    count_df = pd.DataFrame({f'{select_catcol[0]}': s.index, 'Count': s.values})
    st.write(count_df)
    trace = go.Bar(x=s.index,y=s.values,showlegend = False, text=s.values)
    layout = go.Layout(title = f"Bar plot for {select_catcol[0]}")
    data = [trace]
    fig = go.Figure(data=data,layout=layout)
    st.plotly_chart(fig)
else:
    st.info('Please select a categorical column using the dropdown above.')


select_numcol=st.multiselect('Please select numerical column to make  a histogram plot:',dfs.select_dtypes(include='number').columns)




#select_numcol=st.multiselect('Select numerical column:',dfs.select_dtypes(include='number').columns)
#select_numcol=st.multiselect('Select numerical column:',dfs.columns)

if select_numcol:
   st.write("Selected numerical column is : ", select_numcol[0])
   fig, ax = plt.subplots()
   q1 = dfs[select_numcol[0]].quantile(0.25)
   q3 = dfs[select_numcol[0]].quantile(0.75)
   iqr = q3 - q1
   bin_width = (2 * iqr) / (len(dfs[select_numcol[0]]) ** (1 / 3))
   bin_count = int(np.ceil((dfs[select_numcol[0]].max() - dfs[select_numcol[0]].min()) / bin_width))
   selected_value = st.slider('Select number of bins:', min_value=0, max_value=100, value=bin_count)

# Display the selected value
   st.write(f'Select number of bins: {selected_value}')
   ax.hist(dfs[select_numcol[0]], bins=selected_value, edgecolor='black')
   st.pyplot(fig)
 
else:
    st.info('Please select a numerical column using the dropdown above.')


select_catcol=st.multiselect('Please select categorical column to make  a box plot:',dfs.select_dtypes(include='object').columns)
select_numcol=st.multiselect('Please select categorical column to make  a box plot:',dfs.select_dtypes(include='number').columns)

def t_test(sample1, sample2):
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)
    return t_statistic, p_value

def mann_whitney_test(sample1, sample2):
    u_statistic, p_value = stats.mannwhitneyu(sample1, sample2)
    return u_statistic, p_value

if select_catcol and select_numcol:
   dfs[select_numcol[0]]=dfs[select_numcol[0]].fillna(0)
   st.write("Selected categorical column is : ", select_catcol[0], "Selected numerical column is : ",select_numcol[0] )
   fig, ax = plt.subplots()
   df.boxplot(by=select_catcol[0], column=select_numcol[0], ax=ax)
   plt.title(' ')
   plt.xlabel(f'{select_catcol[0]}')
   plt.ylabel(f'{select_numcol[0]}')

# Display the plot in Streamlit app
   st.pyplot(fig)
   st.write(dfs.loc[dfs[select_catcol[0]]==dfs[select_catcol[0]].value_counts().index.tolist()[0]]['Sex'])
   
   # Perform Mann-Whitney U test
   sample1=dfs.loc[dfs[select_catcol[0]]==dfs[select_catcol[0]].value_counts().index.tolist()[0]][select_numcol[0]]
   sample2=dfs.loc[dfs[select_catcol[0]]==dfs[select_catcol[0]].value_counts().index.tolist()[1]][select_numcol[0]]
   # Perform t-test
   t_statistic, t_p_value = t_test(sample1, sample2)

    # Perform Mann-Whitney U test
   u_statistic, u_p_value = mann_whitney_test(sample1, sample2)

    # Display results
   st.write("### Results")
   st.write("#### Independent Samples t-test")
   st.write(f"T-Statistic: {t_statistic}")
   st.write(f"P-Value: {t_p_value}")

   st.write("#### Mann-Whitney U Test")
   st.write(f"U-Statistic: {u_statistic}")
   st.write(f"P-Value: {u_p_value}")



countdfs=dfs['Date of visit'].value_counts().reset_index()
countdfs.columns=['Date of visit','Count']
countdfs['Date of visit'] = pd.to_datetime(countdfs['Date of visit'], errors='coerce', format='%Y-%m-%d')
#countdfs['Date of visit'] = countdfs['Date of visit'].dt.strftime('%Y-%m-%d')

startdate = pd.to_datetime("'2021-01-06'").date()
st.write(countdfs['Date of visit'].values)  

df = pd.DataFrame({'date':countdfs['Date of visit'].values ,'Count':countdfs['Count'].values}, index=countdfs['Date of visit'].values)

st.write(df)

# Sample DataFrame with a datetime column and a group column indicating case or control
df = pd.DataFrame({
    'datetime_column':dfs['Date of visit'],
    'group': dfs['Result 1'],
    'data': dfs['Age']
})

# Streamlit app
df= df.sort_values(by=['datetime_column'])

st.title('Data Plot for Selected Datetime Range')

# Set default start and end dates to the minimum and maximum dates in the DataFrame
start_date = st.date_input("Select start date", min_value=df['datetime_column'].min().date(), max_value=df['datetime_column'].max().date(), value=df['datetime_column'].min().date())
end_date = st.date_input("Select end date", min_value=df['datetime_column'].min().date(), max_value=df['datetime_column'].max().date(), value=df['datetime_column'].max().date())

# Convert to datetime objects
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter the DataFrame based on the selected datetime range
filtered_df = df[(df['datetime_column'].dt.date >= start_date.date()) & (df['datetime_column'].dt.date <= end_date.date())]

# Plot the filtered data for each group separately
plt.figure(figsize=(10, 6))
for group_name, group_data in filtered_df.groupby('group'):
    plt.plot(group_data['datetime_column'], group_data['data'], label=group_name)

plt.title('Data Plot for Selected Datetime Range')
plt.xlabel('Datetime')
plt.ylabel('Data')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Display plot in Streamlit app
st.pyplot(plt)

# Sample DataFrame with a datetime column
df = pd.DataFrame({
    'datetime_column': countdfs['Date of visit'].values,
    'data':countdfs['Count'].values})
st.write(df)
# Streamlit app
st.title('Data Plot for Selected Datetime Range')

# Set default start and end dates to the minimum and maximum dates in the DataFrame
start_date = st.date_input("Select start date", min_value=df['datetime_column'].min().date(), value=df['datetime_column'].min().date())
end_date = st.date_input("Select end date", min_value=df['datetime_column'].min().date(), value=df['datetime_column'].max().date())

# Convert to datetime objects
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter the DataFrame based on the selected datetime range
filtered_df = df[(df['datetime_column'].dt.date >= start_date.date()) & (df['datetime_column'].dt.date <= end_date.date())]

# Plot the filtered data
plt.figure(figsize=(10, 6))
plt.plot(filtered_df['datetime_column'], filtered_df['data'])
plt.title('Data Plot for Selected Datetime Range')
plt.xlabel('Datetime')
plt.ylabel('Data')
plt.xticks(rotation=45)
plt.grid(True)

# Sample DataFrame with temporal data
data = {
    'Date': pd.date_range(start='2022-01-01', periods=100),
    'Value': [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10] * 10  # Sample data with positive and negative values
}
df = pd.DataFrame(data)

# Streamlit app title
st.title('Average of Positive Values Over Time')

# Calculate average of positive values over time
positive_values = df[df['Value'] > 0]
average_positive = positive_values.groupby(positive_values['Date'].dt.month)['Value'].mean()

# Plot the average of positive values over time
plt.figure(figsize=(10, 6))
plt.plot(average_positive.index, average_positive.values, marker='o')
plt.title('Average of Positive Values Over Time')
plt.xlabel('Month')
plt.ylabel('Average of Positive Values')
plt.grid(True)
# Display plot in Streamlit app
st.pyplot(plt)
