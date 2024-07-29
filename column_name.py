replacements = {
    'hiv_rdt': {1: 'Positive', 2: 'Negative'},
    'malaria_rdt': {1: 'Positive', 2: 'Negative'},
    'hepb_rdt': {1: 'Positive', 2: 'Negative'},
    'hepc_rdt': {1: 'Positive', 2: 'Negative'},
    'syphilis_rdt': {1: 'Positive', 2: 'Negative'},
    'siteregion_crf': {1: 'IKORODU', 4: 'ABAKALIKI', 2: 'OWO', 3: 'IRRUA'},
    'sex_crf': {1: 'Male', 2: 'Female'},
    'sex_rdt': {1: 'Male', 2: 'Female'},
    'clinic_crf': {1: 'ISTH', 2: 'General Hospital Ikorodu', 3: 'AEFUTHA, Abakaliki', 4: 'FMC - Owo', 5: 'Other government hospital', 6: 'Other clinic', 7: 'Other hospital'},
    'unit_crf': {1: 'GOPD', 2: 'Pediatrics', 3: 'A&E', 4: 'Annex', 5: 'Other'},
    'sex_sample': {1: 'Male', 2: 'Female'},
    'lassa_pcr': {0: 'yellow', 1: 'other than yellow'},
    'appearance_urin': {0: 'Clear', 1: 'Cloudy/other than clear'},
    'leukocyte_urin': {0: 'Negative or Normal', 1: 'Trace,1+ or 30', 2: '2+ or moderate', 3: '3+ or large'},
    'nitrite_urin': {0: 'Negative or Normal', 1: 'Positive'},
    'protein_urin': {0: 'Negative or Normal', 1: 'Trace, 1+ or 30', 2: '2+ or 100', 3: '3+ or 300', 4: '4+ or 500'},
    'blood_urin': {0: 'Negative or Normal', 1: 'Trace [hemolysis]', 2: '+25 [hemolysis]', 3: '++80 [hemolysis]', 4: '+++200 [hemolysis]', 5: '+10 [non-hemolysis]', 6: '++80 [non-hemolysis]'},
    'ketones_urin': {0: 'Negative or Normal', 1: 'Trace, 1+ small', 2: '2+, moderate, 40', 3: '3+, large, 80', 4: '4+, 160'},
    'bilirubin_urin': {0: 'Negative or Normal', 1: 'Trace, 1+ small', 2: '2+, moderate, 40', 3: '3+, large'},
    'glucose_urin': {0: 'Negative or Normal', 1: 'Trace, 1+, 50 to 150', 2: '2+ or 500', 3: '3+ or 100', 4: '4+ or 2000'},
    'bloodsample': {1: 'Yes', 0: 'No'},
    'urinesample': {1: 'Yes', 0: 'No'},
    'nasosample': {1: 'Yes', 0: 'No'},
    'salivasample': {1: 'Yes', 0: 'No'},
    'oralsample': {1: 'Yes', 0: 'No'},
    'pregnancy': {1: 'Pregnant', 2: 'Not pregnant'},
    'yellowfever_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'lassa_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'ebola_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'marburg_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'westnile_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'zika_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'cchf_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'riftvalley_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'dengue_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'ony_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'covid_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'mpox_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'other1_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'other2_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'other3_pcr': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'syphilis_rdt': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'covid_rdt': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'other1_rdt': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'other2_rdt': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'other3_rdt': {1: 'Positive', 2: 'Negative', 3: 'Pending results', 4: 'Not run'},
    'watersource': {1: 'Stream', 2: 'Dam', 3: 'Well', 4: 'Borehole', 5: ' Pipeborne water', 6: 'Water tanker', 7: 'Sachet water', 8: 'Bottle water', 9: 'Rain water'},
    'housingrisk': {1: 'Riverside', 2: 'Water-logged area', 3: 'Bushy surroundings', 4: 'Farmland settlement', 5: 'Industrial area', 6: 'Around a dumping site', 7: ' Near a bus stop or motor park', 8: 'Near a marketplace'},
    'roofingtype': {0: 'Zinc', 1: 'Aluminum', 2: 'Asbestos', 3: 'Thatched', 4: 'Other'},
    'housingtype': {1: 'Cement', 2: 'Mud', 3: 'Wood', 4: 'Other'},
    'diagnosed_history': {1: 'Yes', 2: 'No', 3: 'Not know'},
    'result_1': {1: 'Positive', 2: 'Negative', 3: 'Inconclusive'},
    'result_2': {1: 'Positive', 2: 'Negative', 3: 'Inconclusive'},
    'result_3': {1: 'Positive', 2: 'Negative', 3: 'Inconclusive'},
    'result_4': {1: 'Positive', 2: 'Negative', 3: 'Inconclusive'},
    'vomiblood_details': {1: 'Bright red', 2: 'Coffee/dark brown'},
    'stoolblood_details': {1: 'Bright red', 2: 'Coffee/dark brown'},
    'urine_details': {1: 'Dark yellow', 2: 'Red', 3: 'Browm'},
    'cough_details': {1: 'with sputum production', 2: 'bloody sputum ', 3: 'Dry'},
    'feverd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'lethargyd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'headached': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'visiond': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'coughd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'jointd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'muscled': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'dyspnoead': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'wheezingd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'eard': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'appetited': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'chestd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'swallowingd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'nausead': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'vomitd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'diarrhead': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'abdominald': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'backd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'hiccupsd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'mouthd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'throatd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'nosed': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'rashd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'seizured': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'bruisingd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'confusiond': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'swellingd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'neckd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'urined': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'bloodd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'stoolbloodd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'nosebloodd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'oralbloodd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'vomitbloodd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'vagbloodd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'intravenousbloodd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'othersymptomsd': {1: '< = 2 days', 2: '3-14 days', 3: '2+ weeks'},
    'ethnicityc_crf': {1: 'Bini', 2: 'Esan', 3: 'Etsako', 4: 'Hausa', 5: 'Igarra', 6: 'Igbo', 7: 'Owan', 8: 'Urhobo', 9: 'Yoruba', 10: 'Other'},
    'community_crf': {1: 'Rural', 2: 'Urban'},
    'state_crf': {1: "Abia", 2: "Adamawa", 3: "Akwa Ibom", 4: "Anambra", 5: "Bauchi", 6: "Bayelsa", 7: "Benue", 8: "Borno", 9: "Cross River", 10: "Delta", 11: "Ebonyi", 12: "Edo", 13: "Ekiti", 14: "Enugu", 15: "FCT", 16: "Gombe", 17: "Imo", 18: "Jigawa", 19: "Kaduna", 20: "Kano", 21: "Katsina", 22: "Kebbi", 23: "Kogi", 24: "Kwara", 25: "Lagos", 26: "Nasarawa", 27: "Niger", 28: "Ogun", 29: "Ondo", 30: "Osun", 31: "Oyo", 32: "Plateau", 33: "Rivers", 34: "Sokoto", 35: "Taraba", 36: "Yobe", 37: "Zamfara", 38: "Other (including other country)"},
    'comorbidity': {1: "Diabetes", 2: "Hypertension", 3: "Chronic cardiac disease", 4: "Chronic pulmonary disease", 5: "Cirrhosis", 6: "Liver diseases", 7: "Stroke", 8: "Solid tumor", 9: "Leukemia or lymphoma", 10: "HIV/AIDS", 11: "Sickle cell anemia", 12: "Rheumatologic disease", 13: "Dementia", 14: "Asplenia", 15: "Tuberculosis", 16: "Hepatitis B infection", 17: "Other", 18: "None/Not applicable"},
    'pregnancy_crf': {1: 'Yes', 0: 'No', 2: 'N/A'},
    'polygamy': {1: 'Yes', 0: 'No', 2: 'N/A'},
    'marital': {1: "Single, never married", 2: "Married or living as a couple", 3: "Divorced/separated"},
    'family_crf': {1: '<=4', 2: '>4'},
    'income_crf': {1: '<= 30,000 naira', 2: '>30,000 naira', 3: 'Not applicable', 4: 'Unknown'},
    'education_crf': {1: 'Informal', 2: 'Primary', 3: 'Secondary', 4: 'Post-secondary diploma', 5: 'Bachelor', 6: 'Higher than bachelor', 7: 'Vocational', 8: 'Not applicable'},
    'occupation2_crf': {1: 'Healthcare worker (government/private)', 2: 'Traditional healer', 3: 'Farmer', 4: 'Business person', 5: 'Livestock trader/butcher', 6: 'Other', 7: 'Not applicable', 8: 'Hunter/trader of game meat', 9: 'Miner/logger', 10: 'Religious leader', 11: 'Housewife', 12: 'Civil servant/government worker', 13: 'Transporter', 14: 'Teacher', 15: 'Student'},
    'occupation1_crf': {1: 'Government/Private employed', 2: 'Self-employed', 3: 'Unemployed', 4: 'Not Applicable'},
    'casecontrol': {1: 'Patient', 2: 'Control'}
}













"""df['sex_crf']=df['sex_crf'].replace({1:'Male',2:'Female'})
df['clinic_crf']=df['clinic_crf'].replace({1:'ISTH',2:'General Hospital Ikorodu',3:'AEFUTHA, Abakaliki',4:'FMC - Owo',5:'Other government hospital',6:'Other clinic',7:'Other hospital'})
df['unit_crf']=df['unit_crf'].replace({1:'GOPD',2:'Pediatrics',3:'A&E',4:'Annex',5:'Other'})
df['sex_sample']=df['sex_sample'].replace({1:'Male',2:'Female'})
df['lassa_pcr']=df['lassa_pcr'].replace({0:'yellow',1:'other than yellow'})
df['appearance_urin']=df['appearance_urin'].replace({0:'Clear',1:'Cloudy/other than clear'})
df['leukocyte_urin']=df['leukocyte_urin'].replace({0:'Negative or Normal',1:'Trace,1+ or 30',2:'2+ or moderate',3:'3+ or large'})
df['nitrite_urin']=df['nitrite_urin'].replace({0:'Negative or Normal',1:'Positive'})
df['protein_urin']=df['protein_urin'].replace({0:'Negative or Normal',1:'Trace, 1+ or 30',2:'2+ or 100',3:'3+ or 300',4:'4+ or 500'})
df['blood_urin']=df['blood_urin'].replace({0:'Negative or Normal',1:'Trace [hemolysis]',2:'+25 [hemolysis]',3:'++80 [hemolysis]',4:'+++200 [hemolysis]',5:'+10 [non-hemolysis]',6:'++80 [non-hemolysis]'})
df['ketones_urin']=df['ketones_urin'].replace({0:'Negative or Normal',1:'Trace, 1+ small',2:'2+, moderate, 40',3:'3+, large, 80',4:'4+, 160'})
df['bilirubin_urin']=df['bilirubin_urin'].replace({0:'Negative or Normal',1:'Trace, 1+ small',2:'2+, moderate, 40',3:'3+, large'})
df['glucose_urin']=df['glucose_urin'].replace({0:'Negative or Normal',1:'Trace, 1+, 50 to 150',2:'2+ or 500',3:'3+ or 100',4:'4+ or 2000'})
df[['bloodsample','urinesample','nasosample','salivasample','oralsample']]=df[['bloodsample','urinesample','nasosample','salivasample','oralsample']].replace({1:'Yes',0:'No'})
df['pregnancy']=df['pregnancy'].replace({1:'Pregnant',2:'Not pregnant'})
df[['yellowfever_pcr','lassa_pcr','ebola_pcr','marburg_pcr','westnile_pcr','zika_pcr','cchf_pcr''riftvalley_pcr','dengue_pcr','ony_pcr','covid_pcr','mpox_pcr','other1_pcr','other2_pcr','other3_pcr']]=df[['yellowfever_pcr','lassa_pcr','ebola_pcr','marburg_pcr','westnile_pcr','zika_pcr','cchf_pcr''riftvalley_pcr','dengue_pcr','ony_pcr','covid_pcr','mpox_pcr','other1_pcr','other2_pcr','other3_pcr']].replace({1:'Positive',2:'Negative',3:'Pending results',4:'Not run'})
df[['syphilis_rdt','covid_rdt','other1_rdt','other2_rdt','other3_rdt']].df[['syphilis_rdt','covid_rdt','other1_rdt','other2_rdt','other3_rdt']].eplace({1:'Positive',2:'Negative',3:'Pending results',4:'Not run'})
df['watersource']=df['watersource'].replace({1:'Stream',2:'Dam',3:'Well',4:'Borehole',5:' Pipeborne water',6:'Water tanker',7:'Sachet water',8:'Bottle water',9:'Rain water'})
df['housingrisk']=df['housingrisk'].replace({1:'Riverside',2:'Water-logged area',3:'Bushy surroundings',4:'Farmland settlement',5:'Industrial area',6:'Around a dumping site',7:' Near a bus stop or motor park',8:'Near a marketplace'})
df['roofingtype']=df['roofingtype'].replace({0:'Zinc',1:'Aluminum',2:'Asbestos',3:'Thatched',4:'Other'})
df['housingtype']=df['housingtype'].replace({1:'Cement',2:'Mud',3:'Wood',4:'Other'})
df['diagnosed_history']=df['diagnosed_history'].replace({1:'Yes',2:'No',3:'Not know'})
df[['result_1','result_2','result_3','result_4']]=df[['result_1','result_2','result_3','result_4']].replace({1:'Positive',2:'Negative',3:'Inconclusive'})
df[['vomiblood_details','stoolblood_details']]=df[['vomiblood_details','stoolblood_details']].replace({1:'Bright red',2:'Coffee/dark brown'})
df['urine_details']=df['urine_details'].replace({1:'Dark yellow',2:'Red',3:'Browm'})
df['cough_details']=df['cough_details'].replace({1:'with sputum production',2:'bloody sputum ',3:'Dry'})
df[["feverd","lethargyd","headached", "visiond", "coughd", "jointd","muscled", "dyspnoead", "wheezingd","eard","appetited", "chestd","swallowingd", "nausead","vomitd","diarrhead","abdominald", "backd","hiccupsd","mouthd","throatd","nosed", "rashd","seizured","bruisingd", "confusiond", "swellingd","neckd","urined","bloodd","stoolbloodd","nosebloodd","oralbloodd","vomitbloodd","vagbloodd","intravenousbloodd","othersymptomsd"]]=df[["feverd","lethargyd","headached", "visiond", "coughd", "jointd","muscled", "dyspnoead", "wheezingd","eard","appetited", "chestd","swallowingd", "nausead","vomitd","diarrhead","abdominald", "backd","hiccupsd","mouthd","throatd","nosed", "rashd","seizured","bruisingd", "confusiond", "swellingd","neckd","urined","bloodd","stoolbloodd","nosebloodd","oralbloodd","vomitbloodd","vagbloodd","intravenousbloodd","othersymptomsd"]].replace({1:'< = 2 days',2:'3-14 days',3:'2+ weeks'})
df['ethnicityc_crf']=df['ethnicityc_crf'].replace({1:'Bini',2:'Esan',3:'Etsako',4:'Hausa',5:'Igarra',6:'Igbo',7:'Owan',8:'Urhobo',9:'Yoruba',10:'Other'})
df['community_crf']=df['community_crf'].replace({1:'Rural',2:'Urban'})
df['state_crf']=df['state_crf'].replace({1: "Abia", 2: "Adamawa", 3: "Akwa Ibom", 4: "Anambra", 5: "Bauchi", 6: "Bayelsa", 7: "Benue", 8: "Borno", 9: "Cross River", 10: "Delta", 11: "Ebonyi", 12: "Edo", 13: "Ekiti", 14: "Enugu", 15: "FCT", 16: "Gombe", 17: "Imo", 18: "Jigawa", 19: "Kaduna", 20: "Kano", 21: "Katsina", 22: "Kebbi", 23: "Kogi", 24: "Kwara", 25: "Lagos", 26: "Nasarawa", 27: "Niger", 28: "Ogun", 29: "Ondo", 30: "Osun", 31: "Oyo", 32: "Plateau", 33: "Rivers", 34: "Sokoto", 35: "Taraba", 36: "Yobe", 37: "Zamfara", 38: "Other (including other country)"})
df['comorbidity']=df['comorbidity'].replace({1: "Diabetes", 2: "Hypertension", 3: "Chronic cardiac disease", 4: "Chronic pulmonary disease", 5: "Cirrhosis", 6: "Liver diseases", 7: "Stroke", 8: "Solid tumor", 9: "Leukemia or lymphoma", 10: "HIV/AIDS", 11: "Sickle cell anemia", 12: "Rheumatologic disease", 13: "Dementia", 14: "Asplenia", 15: "Tuberculosis", 16: "Hepatitis B infection", 17: "Other", 18: "None/Not applicable"})
df[['pregnancy_crf','polygamy']]=df[['pregnancy_crf','polygamy']].replace({1:'Yes',0:'No',2:'N/A'})
df['marital']=df['marital'].replace({1: "Single, never married", 2: "Married or living as a couple", 3: "Divorced/separated"})
df['family_crf']=df['family_crf'].replace({1:'<=4',2:'>4'})
df['income_crf']=df['income_crf'].replace({1: "<= 30,000 naira", 2: ">30,000 naira", 3: "Not applicable", 4: "Unknown"})
df['education_crf']=df['education_crf'].replace({1: "Informal", 2: "Primary", 3: "Secondary", 4: "Post-secondary diploma", 5: "Bachelor", 6: "Higher than bachelor", 7: "Vocational", 8: "Not applicable"})
df['occupation2_crf']=df['occupation2_crf'].replace({1: "Healthcare worker (government/private)", 2: "Traditional healer", 3: "Farmer", 4: "Business person", 5: "Livestock trader/butcher", 8: "Hunter/trader of game meat", 9: "Miner/logger", 10: "Religious leader", 11: "Housewife", 12: "Civil servant/government worker", 13: "Transporter", 14: "Teacher", 15: "Student", 6: "Other", 7: "Not applicable"})
df['occupation1_crf']=df['occupation1_crf'].replace({1: "Government/Private employed", 2: "Self-employed", 3: "Unemployed", 4: "Not Applicable"})
df['casecontrol']=df['casecontrol'].replace({1:'Patient',2:'Control'})"""
