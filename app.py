
from flask import Flask, render_template, request, send_file
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import norm, chi2, kstest, anderson, shapiro
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVR as supportVectorRegressor
from sklearn.ensemble import RandomForestRegressor as randomForestRegressor 
from sklearn.neighbors import KNeighborsRegressor as kNeighborsRegressor
from xgboost import XGBRegressor as xGBRegressor
from sklearn.svm import SVC as supportVectorClassifier  
from sklearn.neighbors import KNeighborsClassifier as kNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as randomForestClassifier
from xgboost import XGBClassifier as xGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import math
import os



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    
    fileWrite=open("ml-code.py", "w")
    fileWrite.write("############################################\n")
    fileWrite.write("#*********Import Basic libraries*********\n")
    fileWrite.write("#############################################\n")
    fileWrite.write("import numpy as np\n")
    fileWrite.write("import pandas as pd\n")
    fileWrite.write("import matplotlib.pyplot as plt\n")
    fileWrite.write("import seaborn as sns\n")
    fileWrite.write("import math\n\n\n")
    
    fileWrite.write("############################################\n")
    fileWrite.write("#*********Getting the dataset*********\n")
    fileWrite.write("#############################################\n")
    
    file_option = request.form.get('file_option')
    if file_option == 'upload':
        file = request.files['file']
        tmpDF = pd.read_csv(file)
        fileWrite.write(f'tmpDF=pd.read_csv({file})\n')
        fileWrite.write("print(tmpDF)\n\n\n")
        
    else:
        selected_option2 = request.form.get('existing_file')
        
        if selected_option2 == 'Classification diabetes':
            file_path = os.path.join(app.root_path, "classificatin dabetes.csv")
            tmpDF = pd.read_csv(file_path)
            fileWrite.write('tmpDF = pd.read_csv("Classification diabetes.csv")\n')
            fileWrite.write("print(tmpDF)\n\n\n")
        elif selected_option2 == 'Regression car data':
            file_path = os.path.join(app.root_path, "regression car data.csv")
            tmpDF = pd.read_csv(file_path)
            fileWrite.write('tmpDF = pd.read_csv("Regression car data.csv")\n')
            fileWrite.write("print(tmpDF)\n\n\n")
        elif selected_option2 == 'Classification Iris':
            file_path = os.path.join(app.root_path, "classification Iris.csv")
            tmpDF = pd.read_csv(file_path)
            fileWrite.write('tmpDF = pd.read_csv("Classification Iris.csv")\n')
            fileWrite.write("print(tmpDF)\n\n\n")
        elif selected_option2 == 'Regression restaurant bill':
            file_path = os.path.join(app.root_path, "regression restaurant_bill.csv")
            tmpDF = pd.read_csv(file_path)
            fileWrite.write('tmpDF = pd.read_csv("Regression restaurant bill.csv")\n')
            fileWrite.write("print(tmpDF)\n\n\n")
        else:
            pass
    
    
    fileWrite.write("############################################\n")
    fileWrite.write("#********Gathering some information*********\n")
    fileWrite.write("############################################\n")
    fileWrite.write("print(tmpDF.head())\n\n")
    fileWrite.write("print(tmpDF.tail())\n\n")
    fileWrite.write("print(tmpDF.shape)\n\n")
    fileWrite.write("print(tmpDF.columns)\n\n")
    fileWrite.write("print(tmpDF.info())\n\n")
    fileWrite.write("print(tmpDF.isnull().sum()/tmpDF.shape[0]*100)\n\n")
    fileWrite.write("print(tmpDF.isnull().sum())\n\n")
    fileWrite.write('#Performing pandas profiling\n')
    fileWrite.write('from pandas_profiling import ProfileReport\n')
    fileWrite.write('prof=ProfileReport(tmpDF)\n')
    fileWrite.write('prof.to_file(output_file="profile_pandas.html)\n\n\n')
    
    fileWrite.write("############################################\n")
    fileWrite.write("#**********Taking all the inputs***********\n")
    fileWrite.write("############################################\n\n")
    target_col_name = request.form['target_col_name']
    fileWrite.write(f'target_col_name="{target_col_name}"\n\n')
    drop_col_nm= request.form['drop_col_nm'].split(",")
    fileWrite.write(f'drop_col_nm={drop_col_nm}\n\n')
    dateTimeColsNames = request.form['dateTimeColsNames'].split(",")
    fileWrite.write(f'dateTimeColsNames={dateTimeColsNames}\n\n')
    drop_nan_col_bound = request.form['drop_nan_col_bound']
    fileWrite.write(f'drop_nan_col_bound={drop_nan_col_bound}\n\n')
    num_col_nm_strip=request.form['num_col_nm_strip'].split(",")
    fileWrite.write(f'num_col_nm_strip={num_col_nm_strip}\n\n')
    num_col_nm_strip_2=request.form['num_col_nm_strip_2'].split(",")
    fileWrite.write(f'num_col_nm_strip_2={num_col_nm_strip_2}\n\n')
    rank_col=request.form['rank_col'].split(",")
    fileWrite.write(f'rank_col={rank_col}\n\n')
    columns_to_split=request.form['columns_to_split'].split(",")
    fileWrite.write(f'columns_to_split={columns_to_split}\n\n')
    delimiters=request.form['delimiters'].split("|-|")
    fileWrite.write(f'delimiters={delimiters}\n\n')
    makedecision=request.form['too_many_unique_values']
    fileWrite.write(f'makedecision="{makedecision}"\n\n\n')
    additional_options = request.form.getlist('additional_options[]')
    

    
    
    
    ######################################################################
    ##################### Dropping columns ############################### 
    ######################################################################
    if drop_col_nm[0].lower()=='none' :
        pass
    else:
        tmpDF.drop(list(drop_col_nm), axis=1, inplace=True)
        fileWrite.write("############################################\n")
        fileWrite.write("#**********Dropping the column***********\n")
        fileWrite.write("############################################\n\n")
        fileWrite.write(f"tmpDF.drop({list(drop_col_nm)}, axis=1, inplace=True)\n")
        fileWrite.write("tmpDF\n\n\n")
        
    
    
    
    ######################################################################
    ##################### Renaming columns ############################### 
    ######################################################################
    
    ##Getting the columns names not present in the drop list###########
    try:
        ###################################################################
        ##Getting the columns names not present in the drop list###########
        ###################################################################

        all_cols=[]
        all_cols_2=[]
        all_cols_3=[]
        all_col_nm_list=list(tmpDF.columns)
        for i in range(len(all_col_nm_list)):
            if all_col_nm_list[i] not in drop_col_nm :
                all_cols.append(all_col_nm_list[i])
                all_cols_2.append(all_col_nm_list[i])
                all_cols_3.append(all_col_nm_list[i])    
            else:
                pass

        for i in range(len(all_cols)):
            sym=[]
            for j in all_cols[i]:
                for k in j:
                    if k=="0" or k=="1" or k=="2" or k=="3" or k=="4" or k=="5" or k=="6" or k=="7" or k=="8" or k=="9" or k==" ":
                            pass
                    elif k.lower()=="a" or k.lower()=="b" or k.lower()=="c" or k.lower()=="d" or k.lower()=="e" or k.lower()=="f" or k.lower()=="g" or k.lower()=="h" or k.lower()=="i" or k.lower()=="j" or k.lower()=="k" or k.lower()=="m" or k.lower()=="n" or k.lower()=="l" or k.lower()=="o" or k.lower()=="p" or k.lower()=="q" or k.lower()=="r" or k.lower()=="s" or k.lower()=="t" or k.lower()=="u" or k.lower()=="v" or k.lower()=="w" or k.lower()=="x" or k.lower()=="y" or k.lower()=="z" or k=="_":
                        pass
                    else:
                        sym.append(k)

            if len(sym)==0:
                pass
            else:
                for l in sym:
                    all_cols[i]=all_cols[i].replace(l,"")

            if   " " in all_cols[i]: 
                all_cols[i]=all_cols[i].replace(" ","_")

            else:
                pass
            del sym
        n=0
        l1=[]
        l2=[]
        if all_cols_2[n]==all_cols[n]:
            all_cols.remove(all_cols[n])
            all_cols_2.remove(all_cols_2[n])
        else:
            pass

        n=n+1
        new_col_nm = {}
        for i in range(len(all_cols)):
            if all_cols_2[i] == all_cols[i]:
                pass
            else:
                new_col_nm[all_cols_2[i]] = all_cols[i]
        if len(new_col_nm)==0:
            pass
        else:
            tmpDF = tmpDF.rename(columns=new_col_nm)
            fileWrite.write("############################################\n")
            fileWrite.write("#*************Renaming columns**************\n")
            fileWrite.write("############################################\n\n")
            fileWrite.write(f"tmpDF=tmpDF.rename(columns={new_col_nm})\n\n")

        tmpDFNewColl=list(tmpDF.columns)
        if dateTimeColsNames[0].lower()=="none":
            pass
        else:

            dl=[]
            for k in range(len(tmpDFNewColl)):
                if all_cols_3[k] in dateTimeColsNames:
                    dl.append(tmpDFNewColl[k])

                else:
                    pass
            if all(x == y for x, y in zip(dl, dateTimeColsNames)):
                pass
            else:
                dateTimeColsNames.clear()  
                dateTimeColsNames.extend(dl)
                fileWrite.write(f"dl={dl}\n")
                fileWrite.write(f"dateTimeColsNames.clear()\n")
                fileWrite.write(f"dateTimeColsNames.extend(dl)\n\n")
            del dl


        if num_col_nm_strip[0].lower()=="none":
            pass
        else:

            nl=[]
            for k in range(len(tmpDFNewColl)):
                if all_cols_3[k] in num_col_nm_strip:
                    nl.append(tmpDFNewColl[k])

                else:
                    pass
            if all(x == y for x, y in zip(nl, num_col_nm_strip)):
                pass
            else:
                num_col_nm_strip.clear()  
                num_col_nm_strip.extend(nl)
                fileWrite.write(f"nl={nl}\n")
                fileWrite.write(f"num_col_nm_strip.clear()\n")
                fileWrite.write(f"num_col_nm_strip.extend(nl)\n\n")


        if num_col_nm_strip_2[0].lower()=="none":
            pass
        else:

            cl=[]
            for k in range(len(tmpDFNewColl)):
                if all_cols_3[k] in num_col_nm_strip_2:
                    cl.append(tmpDFNewColl[k])

                else:
                    pass
            if all(x == y for x, y in zip(cl, num_col_nm_strip_2)):
                pass
            else:
                num_col_nm_strip_2.clear()  
                num_col_nm_strip_2.extend(cl)
                fileWrite.write(f"cl={cl}\n")
                fileWrite.write(f"num_col_nm_strip_2.clear()\n")
                fileWrite.write(f"num_col_nm_strip_2.extend(cl)\n\n")


        if columns_to_split[0].lower()=="none":
            pass
        else:

            sl=[]
            for k in range(len(tmpDFNewColl)):
                if all_cols_3[k] in columns_to_split:
                    sl.append(tmpDFNewColl[k])

                else:
                    pass
            if all(x == y for x, y in zip(sl, columns_to_split)):
                pass
            else:
                columns_to_split.clear()  
                columns_to_split.extend(sl)
                fileWrite.write(f"sl={sl}\n")
                fileWrite.write(f"columns_to_split.clear()\n")
                fileWrite.write(f"columns_to_split.extend(sl)\n\n")


        if rank_col[0].lower()=="none":
            pass
        else:

            rl=[]
            for k in range(len(tmpDFNewColl)):
                if all_cols_3[k] in rank_col:
                    rl.append(tmpDFNewColl[k])

                else:
                    pass
            if all(x == y for x, y in zip(rl, rank_col)):
                pass
            else:
                rank_col.clear()  
                rank_col.extend(rl)
                fileWrite.write(f"rl={rl}\n")
                fileWrite.write(f"rank_col.clear()\n")
                fileWrite.write(f"rank_col.extend(rl)\n\n")



        tarl=[]
        lo=[] 
        if target_col_name in new_col_nm:
            pass
        else:
            lo.append(1)
        if len(new_col_nm)==0:
            pass
        elif len(new_col_nm)!=0 and len(lo)>0:
            for k in range(len(tmpDFNewColl)):
                if all_cols_3[k] == target_col_name:
                    tarl.append(tmpDFNewColl[k])

                else:
                    pass
        else:
            pass
        if len(tarl)==0:
            pass
        else:
            target_col_name=tarl[0]  
            fileWrite.write(f"tarl={tarl}\n")
            fileWrite.write(f"target_col_name=tarl[0] \n\n\n")

    except Exception as e:
        pass
    
    
    ###################################################################
    ########################## Splitting column #######################
    ###################################################################
    
    c=[]
    try:
        

        if columns_to_split[0].lower()=="none":
            pass
        else:
            fileWrite.write("############################################\n")
            fileWrite.write("#*************Splitting column**************\n")
            fileWrite.write("############################################\n\n")


            
            for i, column_name in enumerate(columns_to_split):
                # split the column and add the resulting columns to the DataFrame
                delimiter = delimiters[i]
                if delimiter.lower()=="space":
                    split_columns = tmpDF[column_name].str.split(" ", expand=True)
                    num_columns = len(split_columns.columns)
                    new_column_names = [f'{column_name}_{j+1}' for j in range(num_columns)]
                    tmpDF[new_column_names] = split_columns
                    fileWrite.write(f'tmpDF[{new_column_names}] = tmpDF["{column_name}"].str.split(" ", expand=True)\n')

                    # remove the original column
                    tmpDF.drop(columns=column_name, inplace=True)
                    fileWrite.write(f'tmpDF.drop(columns="{column_name}",inplace=True)\n\n\n')
                else:
                    split_columns = tmpDF[column_name].str.split(delimiter, expand=True)
                    num_columns = len(split_columns.columns)
                    new_column_names = [f'{column_name}_{j+1}' for j in range(num_columns)]
                    tmpDF[new_column_names] = split_columns
                    fileWrite.write(f'tmpDF[{new_column_names}] = tmpDF["{column_name}"].str.split("{delimiter}", expand=True)\n')

                    tmpDF.drop(columns=column_name, inplace=True)
                    fileWrite.write(f'tmpDF.drop(columns="{column_name}",inplace=True)\n\n\n')
                for o in new_column_names:
                    c.append(o)
    except Exception as e:
        pass
    
    if len(c)>0:
        for i in columns_to_split:
            if i in num_col_nm_strip:
                num_col_nm_strip.remove(i)
            elif i in num_col_nm_strip_2:
                num_col_nm_strip_2.remove(i)
            else:
                pass
        for t in c:
            dt=[]
            for p in tmpDF[t]:

                if len(dt)>5:
                    break
                else:
                    if "," in p:
                        v=p.replace(",","")
                        if v.isdigit() or v[0] == '-' and v[1:].isdigit() or v.replace('.', '', 1).isdigit() :
                            dt.append("T")
                        else:
                            dt.append("F")
                    else:

                        if p.isdigit() or p[0] == '-' and p[1:].isdigit() or p.replace('.', '', 1).isdigit():
                            dt.append("T")
                        else:
                            dt.append("F")
            
            if "F" in dt:
                num_col_nm_strip_2.append(t)
            else:
                num_col_nm_strip.append(t)
            dt.clear() 
    else:
        pass
    
    
    
    ###################################################################
    #################Handling datetime column##########################
    ###################################################################
    if "option1" in additional_options:
        try:
            if dateTimeColsNames[0].lower()=="none":
                pass
            else:
                ###################################################################
                ##Getting the columns names not present in the drop list###########
                ###################################################################
                nm=[]


                for i in range(len(dateTimeColsNames)):
                    if dateTimeColsNames[i] in tmpDF.columns:
                        nm.append(dateTimeColsNames[i])
                    else:
                        pass


                ###########################################################################
                ################ cleaning the data#########################################
                ###########################################################################

                if dateTimeColsNames[0].lower()=="none"  or len(nm)==0:
                    pass
                else:
                    fileWrite.write("###########################################################################\n")
                    fileWrite.write("#*********************cleaning date time column****************************\n")
                    fileWrite.write("###########################################################################\n\n")
                    x=[]
                    for i in range(len(nm)):

                        o=tmpDF[nm[i]][0]
                        x.append(o.split(" "))
                        del o


                    for j in range(len(x)):
                        b=x[j]
                        if len(x[j])==1:

                            if b[0][-3]==":" and len(b[0])==5:
                                tmpDF[[ nm[j]+"_"+"hour", nm[j]+"_"+"minute"]] = tmpDF[nm[j]].str.split(':', expand=True)
                                
                                fileWrite.write(f'tmpDF[[ "{nm[j]}"+"_"+"hour", "{nm[j]}"+"_"+"minute"]] = tmpDF["{nm[j]}"].str.split(":", expand=True)\n\n')
                            elif b[0][-3]==":" and b[0][2]==":":
                                tmpDF[[ nm[j]+"_"+"hour", nm[j]+"_"+"minute",nm[j]+"_"+"second"]] = tmpDF[nm[j]].str.split(':', expand=True)
                               
                                fileWrite.write(f'tmpDF[[ "{nm[j]}"+"_"+"hour", "{nm[j]}"+"_"+"minute","{nm[j]}"+"_"+"second"]] = tmpDF["{nm[j]}"].str.split(":", expand=True)\n\n')
                            elif "-" in b[0][0:3] or "/" in b[0][0:3]:
                                tmpDF[nm[j]]=pd.to_datetime(tmpDF[nm[j]])
                                tmpDF[nm[j]+"_"+"year"] = tmpDF[nm[j]].dt.year
                                tmpDF[nm[j]+"_"+"month"] = tmpDF[nm[j]].dt.month
                                tmpDF[nm[j]+"_"+"day"] = tmpDF[nm[j]].dt.day
                                
                                fileWrite.write(f'tmpDF["{nm[j]}"]=pd.to_datetime(tmpDF["{nm[j]}"])\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"year"] = tmpDF["{nm[j]}"].dt.year\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"month"] = tmpDF["{nm[j]}"].dt.month\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"day"] = tmpDF["{nm[j]}"].dt.day\n\n')

                            else:
                                pass

                        elif len(x[j])==2:
                            if len(b[1])==2:
                                tmpDF[ nm[j]] = tmpDF[ nm[j]].str.replace('AM', "")
                                tmpDF[ nm[j]] = tmpDF[ nm[j]].str.replace('PM', "")
                                tmpDF[[ nm[j]+"_"+"hour", nm[j]+"_"+"minute"]] = tmpDF[nm[j]].str.split(':', expand=True)
                               
                                fileWrite.write(f'tmpDF[ "{nm[j]}"] = tmpDF[ "{nm[j]}"].str.replace("AM", "")\n')
                                fileWrite.write(f'tmpDF[ "{nm[j]}"] = tmpDF[ "{nm[j]}"].str.replace("PM", "")\n')
                                fileWrite.write(f'tmpDF[[ "{nm[j]}"+"_"+"hour", "{nm[j]}"+"_"+"minute"]] = tmpDF["{nm[j]}"].str.split(":", expand=True)\n\n')
                            elif b[1][-3]==":" and b[0][2]==":" or b[0][2]=="-":
                                tmpDF[nm[j]]=pd.to_datetime(tmpDF[nm[j]])
                                tmpDF[nm[j]+"_"+"year"] = tmpDF[nm[j]].dt.year
                                tmpDF[nm[j]+"_"+"month"] = tmpDF[nm[j]].dt.month
                                tmpDF[nm[j]+"_"+"day"] = tmpDF[nm[j]].dt.day
                                tmpDF[nm[j]+"_"+"Hour"] = tmpDF[nm[j]].dt.hour
                                tmpDF[nm[j]+"_"+"Minute"] = tmpDF[nm[j]].dt.minute
                                tmpDF[nm[j]+"_"+"Second"] = tmpDF[nm[j]].dt.second
                                
                                fileWrite.write(f'tmpDF["{nm[j]}"]=pd.to_datetime(tmpDF["{nm[j]}"])\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"year"] = tmpDF["{nm[j]}"].dt.year\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"month"] = tmpDF["{nm[j]}"].dt.month\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"day"] = tmpDF["{nm[j]}"].dt.day\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"Hour"] = tmpDF["{nm[j]}"].dt.hour\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"Minute"] = tmpDF["{nm[j]}"].dt.minute\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"Second"] = tmpDF["{nm[j]}"].dt.second\n\n')

                            else:
                                pass
                        elif len(x[j])==3:
                            if len(b[2])==2:
                                tmpDF[nm[j]]=pd.to_datetime(tmpDF[nm[j]])
                                tmpDF[nm[j]+"_"+"year"] = tmpDF[nm[j]].dt.year
                                tmpDF[nm[j]+"_"+"month"] = tmpDF[nm[j]].dt.month
                                tmpDF[nm[j]+"_"+"day"] = tmpDF[nm[j]].dt.day
                                tmpDF[nm[j]+"_"+"Hour"] = tmpDF[nm[j]].dt.hour
                                tmpDF[nm[j]+"_"+"Minute"] = tmpDF[nm[j]].dt.minute
                                tmpDF[nm[j]+"_"+"Second"] = tmpDF[nm[j]].dt.second
                                
                                fileWrite.write(f'tmpDF["{nm[j]}"]=pd.to_datetime(tmpDF["{nm[j]}"])\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"year"] = tmpDF["{nm[j]}"].dt.year\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"month"] = tmpDF["{nm[j]}"].dt.month\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"day"] = tmpDF["{nm[j]}"].dt.day\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"Hour"] = tmpDF["{nm[j]}"].dt.hour\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"Minute"] = tmpDF["{nm[j]}"].dt.minute\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"Second"] = tmpDF["{nm[j]}"].dt.second\n\n')
                            elif b[0][0]=="J" or  b[0][0]=="F" or b[0][0]=="M" or b[0][0]=="A" or b[0][0]=="S" or b[0][0]=="O" or b[0][0]=="N" or b[0][0]=="D":
                                tmpDF[nm[j]]=pd.to_datetime(tmpDF[nm[j]])
                                tmpDF[nm[j]] =tmpDF[nm[j]].dt.strftime('%Y/%m/%d')
                                tmpDF[nm[j]]=pd.to_datetime(tmpDF[nm[j]])
                                tmpDF[nm[j]+"_"+"year"] = tmpDF[nm[j]].dt.year
                                tmpDF[nm[j]+"_"+"month"] = tmpDF[nm[j]].dt.month
                                tmpDF[nm[j]+"_"+"day"] = tmpDF[nm[j]].dt.day
                               
                                fileWrite.write(f'tmpDF["{nm[j]}"]=pd.to_datetime(tmpDF["{nm[j]}"])\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"] =tmpDF["{nm[j]}"].dt.strftime("%Y/%m/%d")\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"]=pd.to_datetime(tmpDF["{nm[j]}"])\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"year"] = tmpDF["{nm[j]}"].dt.year\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"month"] = tmpDF["{nm[j]}"].dt.month\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"day"] = tmpDF["{nm[j]}"].dt.day\n\n')
                            elif b[1][0]=="J" or  b[1][0]=="F" or b[1][0]=="M" or b[1][0]=="A" or b[1][0]=="S" or b[1][0]=="O" or b[1][0]=="N" or b[1][0]=="D":
                                tmpDF[nm[j]]=pd.to_datetime(tmpDF[nm[j]])
                                tmpDF[nm[j]] =tmpDF[nm[j]].dt.strftime('%Y/%m/%d')
                                tmpDF[nm[j]]=pd.to_datetime(tmpDF[nm[j]])
                                tmpDF[nm[j]+"_"+"year"] = tmpDF[nm[j]].dt.year
                                tmpDF[nm[j]+"_"+"month"] = tmpDF[nm[j]].dt.month
                                tmpDF[nm[j]+"_"+"day"] = tmpDF[nm[j]].dt.day
                                
                                fileWrite.write(f'tmpDF["{nm[j]}"]=pd.to_datetime(tmpDF["{nm[j]}"])\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"] =tmpDF["{nm[j]}"].dt.strftime("%Y/%m/%d")\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"]=pd.to_datetime(tmpDF["{nm[j]}"])\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"year"] = tmpDF["{nm[j]}"].dt.year\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"month"] = tmpDF["{nm[j]}"].dt.month\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"day"] = tmpDF["{nm[j]}"].dt.day\n\n')
                            else:
                                pass 
                        elif len(x[j])==4:
                            if b[0][0]=="S"  or b[0][0]=="M" or b[0][0]=="T" or b[0][0]=="W":
                                tmpDF[nm[j]]=pd.to_datetime(tmpDF[nm[j]])
                                tmpDF[nm[j]] =tmpDF[nm[j]].dt.strftime('%a, %b %d %Y')
                                tmpDF[nm[j]]=pd.to_datetime(tmpDF[nm[j]])
                                tmpDF[nm[j]+"_"+"year"] = tmpDF[nm[j]].dt.year
                                tmpDF[nm[j]+"_"+"month"] = tmpDF[nm[j]].dt.month
                                tmpDF[nm[j]+"_"+"day"] = tmpDF[nm[j]].dt.day
                                
                                fileWrite.write(f'tmpDF["{nm[j]}"]=pd.to_datetime(tmpDF["{nm[j]}"])\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"] =tmpDF["{nm[j]}"].dt.strftime("%a, %b %d %Y")\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"]=pd.to_datetime(tmpDF["{nm[j]}"])\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"year"] = tmpDF["{nm[j]}"].dt.year\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"month"] = tmpDF["{nm[j]}"].dt.month\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"day"] = tmpDF["{nm[j]}"].dt.day\n\n')

                            elif b[2][0]=="S"  or b[2][0]=="M" or b[2][0]=="T" or b[2][0]=="W":
                                tmpDF[nm[j]]=pd.to_datetime(tmpDF[nm[j]])
                                tmpDF[nm[j]] =tmpDF[nm[j]].dt.strftime('%d %b, %a %Y')
                                tmpDF[nm[j]]=pd.to_datetime(tmpDF[nm[j]])
                                tmpDF[nm[j]+"_"+"year"] = tmpDF[nm[j]].dt.year
                                tmpDF[nm[j]+"_"+"month"] = tmpDF[nm[j]].dt.month
                                tmpDF[nm[j]+"_"+"day"] = tmpDF[nm[j]].dt.day
                                
                                fileWrite.write(f'tmpDF["{nm[j]}"]=pd.to_datetime(tmpDF["{nm[j]}"])\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"] =tmpDF["{nm[j]}"].dt.strftime("%d %b, %a %Y")\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"]=pd.to_datetime(tmpDF["{nm[j]}"])\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"year"] = tmpDF["{nm[j]}"].dt.year\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"month"] = tmpDF["{nm[j]}"].dt.month\n')
                                fileWrite.write(f'tmpDF["{nm[j]}"+"_"+"day"] = tmpDF["{nm[j]}"].dt.day\n\n')
                            else:
                                pass 
                        del b
                    del x
                del nm
                tmpDF.drop(dateTimeColsNames, axis=1,inplace=True)
                fileWrite.write(f'tmpDF.drop({dateTimeColsNames}, axis=1,inplace=True)\n')
                fileWrite.write('\n\n\n') 
        except Exception as e:
            pass

        
        ##################################################################################
        ########################Change into numerical datatype ###########################
        ##################################################################################
        try:
            if num_col_nm_strip[0].lower()!="none":
                fileWrite.write("###############################################################\n")
                fileWrite.write("#*************change into numerical data type******************\n")
                fileWrite.write("###############################################################\n\n")
                ##################################################################################
                #####################################Selecting columns ###########################
                ##################################################################################
                final_numerical_cols_to_strip=[]
                for i in range(len(num_col_nm_strip)):
                    if num_col_nm_strip[i] in tmpDF.columns:
                        final_numerical_cols_to_strip.append(num_col_nm_strip[i])
                    else:
                        pass

                
                ##################################################################################
                #####################################Stripping columns ###########################
                ##################################################################################

                for i in range(len(final_numerical_cols_to_strip)):
                    dict_val={}
                    drop_ll=[]
                    n=0
                    for j in tmpDF[final_numerical_cols_to_strip[i]]:

                        tmp_single_val=""
                        for k in str(j):
                            if k=="0" or k=="1" or k=="2" or k=="3" or k=="4" or k=="5" or k=="6" or k=="7" or k=="8" or k=="9" or k==".":
                                tmp_single_val=tmp_single_val+k
                            else:
                                pass

                        
                        if  pd.isna(j)==True or len(tmp_single_val)==0 or tmp_single_val=="nan" or tmp_single_val=="NaN":
                            pass
                        else:
                            if str(tmp_single_val).count(".")>1 or str(tmp_single_val)[len(tmp_single_val)-1]==".":
                                drop_ll.append(n)
                            elif  str(tmp_single_val)[0]==".":
                                tmp_single_val="0"+""+tmp_single_val
                            else:
                                pass



                        if str(j)==tmp_single_val:
                            pass
                        else:
                            if pd.isna(j)==True:
                                pass
                            elif len(tmp_single_val)==0:
                                dict_val[j]= np.nan
                            else:
                                dict_val[j]=tmp_single_val
                                
                        tmp_single_val=""
                        n=n+1
                    
                    if len(drop_ll)==0:
                        pass
                    else:
                        tmpDF.drop(drop_ll,inplace=True)
                        fileWrite.write(f'tmpDF.drop({drop_ll},inplace=True)\n') 

                    if len(dict_val)==0:
                        pass
                    else:
                        fileWrite.write(f'dict_val={dict_val}\n')    
                        tmpDF[final_numerical_cols_to_strip[i]]=tmpDF[final_numerical_cols_to_strip[i]].replace(dict_val)
                        fileWrite.write(f'tmpDF["{final_numerical_cols_to_strip[i]}"]=tmpDF["{final_numerical_cols_to_strip[i]}"].replace(dict_val)\n')
                    
                    
                    dict_val.clear()
                    drop_ll.clear()
            else:
                pass
            
        except Exception as e:
            pass
        
        final_numerical_cols_to_strip_2=[]
        for t in range(len(num_col_nm_strip)):
            if num_col_nm_strip[t] in tmpDF.columns:
                final_numerical_cols_to_strip_2.append(num_col_nm_strip[t])
            else:
                pass

        

        if len(final_numerical_cols_to_strip_2)==0:
            pass
        else:
            for i in range(len(final_numerical_cols_to_strip_2)):
                vzc=[]
                for q in tmpDF[final_numerical_cols_to_strip_2[i]]:
                    if "." in str(q):
                        vzc.append("y")
                    else:
                        vzc.append("n")
                if "y" in vzc:
                    tmpDF[final_numerical_cols_to_strip_2[i]] = tmpDF[final_numerical_cols_to_strip_2[i]].astype(float)
                    fileWrite.write(f'tmpDF["{final_numerical_cols_to_strip_2[i]}"] = tmpDF["{final_numerical_cols_to_strip_2[i]}"].astype(float)\n\n\n')
                else:
                    has_nan = tmpDF[final_numerical_cols_to_strip_2[i]].isna().any()
                    if has_nan:
                        tmpDF[final_numerical_cols_to_strip_2[i]] = tmpDF[final_numerical_cols_to_strip_2[i]].astype(float)
                        fileWrite.write(f'tmpDF["{final_numerical_cols_to_strip_2[i]}"] = tmpDF["{final_numerical_cols_to_strip_2[i]}"].astype(float)\n\n\n')
                    else:
                        tmpDF[final_numerical_cols_to_strip_2[i]] = tmpDF[final_numerical_cols_to_strip_2[i]].astype(int)
                        fileWrite.write(f'tmpDF["{final_numerical_cols_to_strip_2[i]}"] = tmpDF["{final_numerical_cols_to_strip_2[i]}"].astype(int)\n\n\n')

                vzc.clear()
        
        ##################################################################################
        ##################### Change into categorical datatype  ##########################
        ##################################################################################
        try:
            if num_col_nm_strip_2[0].lower()!="none":
                ##################################################################################
                #####################################Selecting columns ###########################
                ##################################################################################

                final_categorical_cols_to_strip=[]
                for i in range(len(num_col_nm_strip_2)):
                    if num_col_nm_strip_2[i] in tmpDF.columns:
                        final_categorical_cols_to_strip.append(num_col_nm_strip_2[i])
                    else:
                        pass
                

                ##################################################################################
                #####################################Stripping columns ###########################
                ##################################################################################
                for i in range(len(final_categorical_cols_to_strip)):
                    dict_cat_val={}
                    for j in tmpDF[final_categorical_cols_to_strip[i]]:
                        if  pd.isna(j)==True or j=="NaN" or j=="nan":
                            pass
                        else:
                            xv=j.split(" ")
                            tmplist=[]
                            tmplist2=[]
                            for k in xv:
                                tmp_single_val=""
                                for b in k:
                                    n=b.lower()
                                    if n=="a" or n=="b" or n=="c" or n=="d" or n=="e" or n=="f" or n=="g" or n=="h" or n=="i" or n=="j" or n=="k" or n=="m" or n=="n" or n=="l" or n=="o" or n=="p" or n=="q" or n=="r" or n=="s" or n=="t" or n=="u" or n=="v" or n=="w" or n=="x" or n=="y" or n=="z" or n=="0" or n=="1" or n=="2" or n=="3" or n=="4" or n=="5" or n=="6" or n=="7" or n=="8" or n=="9":
                                        tmp_single_val=tmp_single_val+b
                                    else:
                                        pass

                                if len(tmp_single_val)!=0:
                                    tmplist.append(tmp_single_val)
                                else:
                                    pass

                                if len(tmplist)<=5:
                                    tmplist2=tmplist
                                else:
                                    tmplist2=tmplist[0:5]

                            tmp_single_val2=" ".join(tmplist2)

                            if j==tmp_single_val2:
                                pass
                            else:
                                dict_cat_val[j]=tmp_single_val2

                            tmplist2.clear()
                            tmplist.clear()

                    if len(dict_cat_val)==0:
                        pass
                    else:
                        fileWrite.write("###############################################################\n")
                        fileWrite.write("#*************change into categorical data type***************\n")
                        fileWrite.write("###############################################################\n\n")
                        fileWrite.write(f'dict_cat_val={dict_cat_val}\n')    
                        tmpDF[final_categorical_cols_to_strip[i]]=tmpDF[final_categorical_cols_to_strip[i]].replace(dict_cat_val)
                        fileWrite.write(f'tmpDF["{final_categorical_cols_to_strip[i]}"]=tmpDF["{final_categorical_cols_to_strip[i]}"].replace(dict_cat_val)\n\n')


            else:
                pass
        except Exception as e:
            pass
        
        
        #######################################################
        #************ Removing constant features **************
        ########################################################
        
        tmp_colll=tmpDF.drop(target_col_name, axis=1)
        constant_columns = [col for col in tmp_colll if tmpDF[col].nunique() == 1]
        tmpDF.drop(constant_columns, axis=1, inplace=True)
        if len(constant_columns)==0:
            pass
        else:
            fileWrite.write("#################################################\n")
            fileWrite.write("#******** Removing constant features ************\n")
            fileWrite.write("#################################################\n")
            fileWrite.write("tmp_colll=tmpDF.drop(target_col_name, axis=1)\n")
            fileWrite.write("constant_columns = [col for col in tmp_colll if tmpDF[col].nunique() == 1]\n")
            fileWrite.write("tmpDF.drop(constant_columns, axis=1, inplace=True)\n\n\n")
        del tmp_colll
        del constant_columns
        
        ##################################################################################
        ############################ Filling the missing value ###########################
        ##################################################################################
        try:
            fileWrite.write("##########################################################\n")
            fileWrite.write('#************** Filling missing values *********************\n')
            fileWrite.write("###########################################################\n\n")
            ######################################################################
            ###############getting the columns to need to be clean#################
            ######################################################################
            if tmpDF.isna().any().any():
                if tmpDF[target_col_name].isnull().sum()/tmpDF.shape[0]*100>=17:
                    fileWrite.write("'''#Dropping target column row if it's contain missing value more than 17% '''\n")
                    tmpDF.dropna(subset=[target_col_name],inplace=True)
                    fileWrite.write(f'tmpDF.dropna(subset=["{target_col_name}"],inplace=True)\n\n')            

                else:
                    pass

                all_col_nm=list(tmpDF.columns)
                delete_cols=[]
                col_need_to_cln=[]
                remaing_cols_name=[]
                for i in range(len(all_col_nm)):
                    if  tmpDF[all_col_nm[i]].isnull().sum()/tmpDF.shape[0]*100>drop_nan_col_bound:
                        if all_col_nm[i]==target_col_name:
                            col_need_to_cln.append(all_col_nm[i])
                        else:
                            delete_cols.append(all_col_nm[i])
                    elif tmpDF[all_col_nm[i]].isnull().sum()/tmpDF.shape[0]*100>0 and tmpDF[all_col_nm[i]].isnull().sum()/tmpDF.shape[0]*100<drop_nan_col_bound:
                        col_need_to_cln.append(all_col_nm[i])
                    else:
                        remaing_cols_name.append(all_col_nm[i])


                


                if len(delete_cols)==0:
                    pass
                else:
                    fileWrite.write("#******* Dropping columns which contain high percent of nan value *******\n")
                    tmpDF.drop(delete_cols,axis=1,inplace=True)
                    fileWrite.write(f'tmpDF.drop({delete_cols},axis=1,inplace=True)\n')
                    
                categorical_column_1=list(tmpDF.select_dtypes(include="object").columns)
                if len(categorical_column_1)>0:
                    ######################################################################
                    ############selecting the columns to clearning columns#############
                    ######################################################################

                    no_nan_cat_contain=[]
                    nan_cat_contain=[]
                    for i in range(len(categorical_column_1)):
                        if tmpDF[categorical_column_1[i]].isnull().sum()/tmpDF.shape[0]*100==0:
                            no_nan_cat_contain.append(categorical_column_1[i])
                        else:
                            nan_cat_contain.append(categorical_column_1[i])

                    

                    col_name_fill_nan=str(None)
                    col_data_fill_nan=[]
                    if len(no_nan_cat_contain)>0:
                        lk=[]
                        for j in range(len(no_nan_cat_contain)):
                            if len(list(tmpDF[no_nan_cat_contain[j]].unique())) <=15:
                                lk.append(len(list(tmpDF[no_nan_cat_contain[j]].unique())))
                            else:
                                pass
                        
                        lk.sort()
                        
                        if len(lk)==0:
                            pass
                        else:
                            for p in range(len(no_nan_cat_contain)):

                                if lk[len(lk)-1]==len(list(tmpDF[no_nan_cat_contain[p]].unique())):
                                    col_name_fill_nan=no_nan_cat_contain[p]
                                    col_data_fill_nan=list(tmpDF[no_nan_cat_contain[p]].unique())


                                else:
                                    pass
                    
                    else:
                        pass

                    if len(no_nan_cat_contain)==0 or len(col_data_fill_nan)==0:

                        oo=[]
                        for k in range(len(nan_cat_contain)):
                            oo.append(tmpDF[nan_cat_contain[k]].isnull().sum()/tmpDF.shape[0]*100)
                        oo.sort()

                        for l in range(len(nan_cat_contain)):

                            if oo[0]==tmpDF[nan_cat_contain[l]].isnull().sum()/tmpDF.shape[0]*100:

                                col_name_fill_nan=nan_cat_contain[l]
                                kn=tmpDF[nan_cat_contain[l]].fillna(tmpDF[nan_cat_contain[l]].mode()[0])
                                col_data_fill_nan.clear()
                                x=list(kn.unique())

                                col_data_fill_nan=x
                                
                            else:
                                pass
                        
                    fileWrite.write("#**** slecting columns and columns value to clean data ***** \n")
                    fileWrite.write(f'col_name_fill_nan="{col_name_fill_nan}"\n')
                    fileWrite.write(f'col_data_fill_nan={col_data_fill_nan}\n\n')
                    
                    ######################################################################
                    #################### Filling the categorical column#####################
                    ######################################################################
                    Catego_cln_dataFrame=None
                    nan_contain_catego_col=[]        
                    if tmpDF[tmpDF.select_dtypes(include="object").columns].isna().any().any():
                        fileWrite.write('#******* Filling categorical missing data *******\n')


                        for kl in col_need_to_cln:
                            if kl in tmpDF.select_dtypes(include="object").columns:
                                nan_contain_catego_col.append(kl)
                            else:
                                pass

                        fileWrite.write(f'nan_contain_catego_col={nan_contain_catego_col}\n')
                        cleaned_col=[]
                        fileWrite.write('cleaned_col=[]\n')
                        for cat_data_j in range(len(nan_contain_catego_col)):
                            col_unq_val_to_fill_nan=list(col_data_fill_nan)

                            v=0
                            tmp=None
                            for cat_data_i in range(len(col_unq_val_to_fill_nan)):
                                col_unq_val_to_fill_nan_2=col_unq_val_to_fill_nan[v]
                                col_unq_val_to_fill_nan_3=""
                                col_unq_val_to_fill_nan_3=col_unq_val_to_fill_nan_2
                                xx=tmpDF[col_name_fill_nan]==col_unq_val_to_fill_nan_3
                                var_nm=col_unq_val_to_fill_nan_3.replace(" ","_")
                                fileWrite.write(f'{var_nm.upper()}=tmpDF["{col_name_fill_nan}"]=="{col_unq_val_to_fill_nan_3}"\n')
                                xxx=tmpDF.loc[xx,[nan_contain_catego_col[cat_data_j]]]
                                fileWrite.write(f'{var_nm.upper()+"_"+"1"}=tmpDF.loc[{var_nm.upper()},[nan_contain_catego_col[{cat_data_j}]]]\n')

                                xxxx=xxx[[nan_contain_catego_col[cat_data_j]]].mode()
                                fileWrite.write(f'{var_nm.upper()+"_"+"2"}={var_nm.upper()+"_"+"1"}[[nan_contain_catego_col[{cat_data_j}]]].mode()\n')
                                
                                filled_data=xxx[nan_contain_catego_col[cat_data_j]].fillna(xxxx.loc[0][0])
                                fileWrite.write(f'filled_data={var_nm.upper()+"_"+"1"}[nan_contain_catego_col[{cat_data_j}]].fillna({var_nm.upper()+"_"+"2"}.loc[0][0])\n')

                                if  tmp is None:
                                    tmp=filled_data
                                    fileWrite.write('tmp=filled_data\n\n')
                                else:
                                    tmp=pd.concat([tmp,filled_data],ignore_index=False)
                                    fileWrite.write('tmp=pd.concat([tmp,filled_data],ignore_index=False)\n\n')
                                v=v+1
                                
                            x=pd.Series(tmp)
                            fileWrite.write('x=pd.Series(tmp)\n')
                            av=x.sort_index(ascending=True)
                            fileWrite.write('av=x.sort_index(ascending=True)\n')
                            cleaned_col.append(av)
                            fileWrite.write('cleaned_col.append(av)\n\n')

                        dictionay_for_crt_dataFrame={}
                        fileWrite.write('dictionay_for_crt_dataFrame={}\n')
                        for cat_data_k in range(len(cleaned_col)):
                            dictionay_for_crt_dataFrame[nan_contain_catego_col[cat_data_k]]=cleaned_col[cat_data_k]
                            fileWrite.write(f'dictionay_for_crt_dataFrame["{nan_contain_catego_col[cat_data_k]}"]=cleaned_col[{cat_data_k}]\n\n')

                        Catego_cln_dataFrame=pd.DataFrame(dictionay_for_crt_dataFrame)

                        for qp in range(len(nan_contain_catego_col)):
                            if Catego_cln_dataFrame[nan_contain_catego_col[qp]].isnull().sum()>0:
                                Catego_cln_dataFrame[nan_contain_catego_col[qp]].fillna(method="ffill", inplace=True)
                                fileWrite.write(f'Catego_cln_dataFrame["{nan_contain_catego_col[qp]}"].fillna(method="ffill", inplace=True)\n')
                        fileWrite.write('Catego_cln_dataFrame=pd.DataFrame(dictionay_for_crt_dataFrame)\n\n')

                    else:
                        pass



                    ######################################################################
                    #################### Filling the numerical column ####################
                    ######################################################################
                    numeric_cleand_dataframe=None
                    nan_contain_numeric_col=[]        
                    if tmpDF[tmpDF.select_dtypes(include=["int64","float64","int32","float32"]).columns].isna().any().any():
                        fileWrite.write("#******** filling numerical missing data ***********\n")

                        for kll in col_need_to_cln:
                            if kll in tmpDF.select_dtypes(include=["int64","float64","int32","float32"]).columns:
                                nan_contain_numeric_col.append(kll)
                            else:
                                pass


                        fileWrite.write(f'nan_contain_numeric_col={nan_contain_numeric_col}\n')
                        cleaned_num_col=[]
                        fileWrite.write("cleaned_num_col=[]\n")
                        for num_data_j in range(len(nan_contain_numeric_col)):
                            col_unq_val_to_fill_nan_for_nume=list(col_data_fill_nan)
                            v=0
                            tmp=None
                            for num_data_i in range(len(col_unq_val_to_fill_nan_for_nume)):
                                nan_contain_numeric_col_2=col_unq_val_to_fill_nan_for_nume[v]
                                nan_contain_numeric_co_3=""
                                nan_contain_numeric_col_3=nan_contain_numeric_col_2
                                xx=tmpDF[col_name_fill_nan]==nan_contain_numeric_col_3
                                var_nm_num=nan_contain_numeric_col_3.replace(" ","_")
                                fileWrite.write(f'{var_nm_num.upper()}=tmpDF["{col_name_fill_nan}"]=="{nan_contain_numeric_col_3}"\n')
                                xxx=tmpDF.loc[xx,[nan_contain_numeric_col[num_data_j]]]
                                fileWrite.write(f'{var_nm_num.upper()+"_"+"1"}=tmpDF.loc[{var_nm_num.upper()},[nan_contain_numeric_col[{num_data_j}]]]\n')
                                xxxx=xxx[[nan_contain_numeric_col[num_data_j]]].mean()
                                fileWrite.write(f'{var_nm_num.upper()+"_"+"2"}={var_nm_num.upper()+"_"+"1"}[[nan_contain_numeric_col[{num_data_j}]]].mean()\n')

                                filled_num_data=xxx[nan_contain_numeric_col[num_data_j]].fillna(xxxx[0])
                                fileWrite.write(f'filled_num_data={var_nm_num.upper()+"_"+"1"}[nan_contain_numeric_col[{num_data_j}]].fillna({var_nm_num.upper()+"_"+"2"}[0])\n\n')

                                if  tmp is None:
                                    tmp=filled_num_data
                                    fileWrite.write('tmp=filled_num_data\n')
                                else:
                                    tmp=pd.concat([tmp,filled_num_data],ignore_index=False)
                                    fileWrite.write('tmp=pd.concat([tmp,filled_num_data],ignore_index=False)\n')
                                v=v+1


                            x=pd.Series(tmp)
                            fileWrite.write('x=pd.Series(tmp)\n')
                            avx=x.sort_index(ascending=True)
                            fileWrite.write('avx=x.sort_index(ascending=True)\n')
                            cleaned_num_col.append(avx)
                            fileWrite.write('cleaned_num_col.append(avx)\n\n')

                        dictionay_for_crt_num_dataFrame={}
                        fileWrite.write('dictionay_for_crt_num_dataFrame={}\n')
                        for num_data_k in range(len(cleaned_num_col)):
                            dictionay_for_crt_num_dataFrame[nan_contain_numeric_col[num_data_k]]=cleaned_num_col[num_data_k]
                            fileWrite.write(f'dictionay_for_crt_num_dataFrame["{nan_contain_numeric_col[num_data_k]}"]=cleaned_num_col[{num_data_k}]\n\n')
                        numeric_cleand_dataframe=pd.DataFrame(dictionay_for_crt_num_dataFrame)
                        fileWrite.write('numeric_cleand_dataframe=pd.DataFrame(dictionay_for_crt_num_dataFrame)\n\n')

                        
                    ######################################################################
                    ###############Creating the clean dataset########################## 
                    ######################################################################
                    fileWrite.write("#********** Creating clean dataset ***************")



                    remaing_cols=tmpDF[remaing_cols_name]
                    fileWrite.write(f'remaing_cols=tmpDF[{remaing_cols_name}]\n')



                    if len(nan_contain_numeric_col)==0:
                        df=pd.merge(Catego_cln_dataFrame,remaing_cols,right_index=True,left_index=True)
                        fileWrite.write('df=pd.merge(Catego_cln_dataFrame,remaing_cols,right_index=True,left_index=True)\n\n')
                        

                    else:
                        pass


                    if len(nan_contain_catego_col)==0:
                        df=pd.merge(numeric_cleand_dataframe,remaing_cols,right_index=True,left_index=True)
                        fileWrite.write('df=pd.merge(numeric_cleand_dataframe,remaing_cols,right_index=True,left_index=True)\n\n')

                        
                    else:
                        pass


                    if len(nan_contain_catego_col)>0 and len(nan_contain_numeric_col)>0:
                        catago_and_numeric=pd.merge(Catego_cln_dataFrame,numeric_cleand_dataframe,right_index=True,left_index=True)
                        df=pd.merge(catago_and_numeric,remaing_cols,right_index=True,left_index=True)
                        fileWrite.write('catago_and_numeric=pd.merge(Catego_cln_dataFrame,numeric_cleand_dataframe,right_index=True,left_index=True)\n')
                        fileWrite.write('df=pd.merge(catago_and_numeric,remaing_cols,right_index=True,left_index=True)\n\n')

                        

                    else:
                        pass


                elif len(categorical_column_1)==0: 
                    nan_contain_numeric_col=[]
                    numerical_column=tmpDF.select_dtypes(include=["int64","float64","int32","float32"]).columns

                    for get_nan_cont_col in range(len(numerical_column)):
                        nume_col_name_miss_test=tmpDF[numerical_column[get_nan_cont_col]].isnull().sum()
                        if nume_col_name_miss_test>0:
                            nan_contain_numeric_col.append(numerical_column[get_nan_cont_col])
                        else:
                            pass

                    if len(nan_contain_numeric_col)==0:
                        df=tmpDF
                        fileWrite.write('df=tmpDF\n\n')
                    else:
                        from sklearn.impute import KNNImputer
                        imputer = KNNImputer(n_neighbors=len(list(tmpDF.columns)))
                        df_imputed = pd.DataFrame(imputer.fit_transform(tmpDF), columns=tmpDF.columns)
                        df=df_imputed
                        fileWrite.write('from sklearn.impute import KNNImputer\n')
                        fileWrite.write('imputer = KNNImputer(n_neighbors=len(list(tmpDF.columns)))\n')
                        fileWrite.write('df_imputed = pd.DataFrame(imputer.fit_transform(tmpDF), columns=tmpDF.columns)\n')
                        fileWrite.write('df=df_imputed\n\n')



            else:
                df=tmpDF
                fileWrite.write('df=tmpDF\n\n')

            df=df.reset_index(drop=True)
            fileWrite.write('df=df.reset_index(drop=True)\n\n\n')                

            df.head(50)
        except Exception as e:
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=len(list(tmpDF.columns)))
            df_imputed = pd.DataFrame(imputer.fit_transform(tmpDF), columns=tmpDF.columns)
            df=df_imputed
            fileWrite.write('from sklearn.impute import KNNImputer\n')
            fileWrite.write('imputer = KNNImputer(n_neighbors=len(list(tmpDF.columns)))\n')
            fileWrite.write('df_imputed = pd.DataFrame(imputer.fit_transform(tmpDF), columns=tmpDF.columns)\n')
            fileWrite.write('df=df_imputed\n\n')
        
    else:
        df=tmpDF
        fileWrite.write('df=tmpDF\n\n\n')
    
    #######################################################
    #********* Removing Outliers for numerical data *******
    ########################################################
    
    if "option2" in additional_options:
        num_col_for_out_tmp=df.select_dtypes(include=["int64","float64","int32","float32"])
        num_col_for_out=[]
        for j in range(len(num_col_for_out_tmp.columns)):
            if num_col_for_out_tmp.columns[j]!=target_col_name:
                num_col_for_out.append(num_col_for_out_tmp.columns[j])
            else:
                pass
          
        #######################################################
        #********* Removing Outliers EDA distplot *******
        ########################################################  
        fileWrite.write("################################################\n")
        fileWrite.write("#********* EDA for Outliers Handling************\n")
        fileWrite.write("################################################\n\n\n")
        
        fileWrite.write('#******** distplot for see the distribution *******\n') 
        nnn=1
        for j in range(len(num_col_for_out)):
            
            if len(num_col_for_out)==0:
                pass
            else:
                fileWrite.write(f'plt.subplot({math.ceil(len(num_col_for_out)/3)},3,nnn)\n')
                fileWrite.write(f'sns.distplot(df["{num_col_for_out[j]}"])\n')
                fileWrite.write('plt.subplots_adjust(left=0.0,bottom=0.0,right=2.0,top=3.0,wspace=0.4,hspace=0.4)\n\n')
                nnn=nnn+1
        fileWrite.write('\n')
        del nnn 
        #######################################################
        #********* Removing Outliers EDA boxplot *******
        ######################################################## 
        fileWrite.write('#******** Box plot for identify outliers*******\n') 
        
        nnn=1
        for j in range(len(num_col_for_out)):
            
            if len(num_col_for_out)==0:
                pass
            else:
                fileWrite.write(f'plt.subplot({math.ceil(len(num_col_for_out)/3)},3,nnn)\n')
                fileWrite.write(f'sns.boxplot(df["{num_col_for_out[j]}"])\n')
                fileWrite.write('plt.subplots_adjust(left=0.0,bottom=0.0,right=2.0,top=3.0,wspace=0.4,hspace=0.4)\n\n')
                nnn=nnn+1
        fileWrite.write('\n')
        del nnn  
        
        
        
        try:
            fileWrite.write("########################################################\n")
            fileWrite.write("#********* Removing Outliers for numerical data *********\n")
            fileWrite.write("########################################################\n\n")

            kk=[]
            nn=0
            row_to_drp=[]
            for i in range(len(num_col_for_out)):
                
                data=df[num_col_for_out[i]]

                col_nm=num_col_for_out[i]


                fileWrite.write(f'data=df["{num_col_for_out[i]}"]\n')
                
        
                #####################################################################################################################
                # perform the Anderson-Darling test
                result = anderson(data, 'norm')

                # print the test statistic and critical values
                for i in range(len(result.critical_values)):
                    sl, cv = result.significance_level[i], result.critical_values[i]
                    


                # interpret the result
                alpha = 0.05
                if result.statistic < result.critical_values[2]:
                    kk.append("True")
                else:
                    kk.append("False")
                ####################################################################################################################
                # perform the Kolmogorov-Smirnov test
                test_statistic, p_value = kstest(data, 'norm')

                alpha = 0.05
                if p_value > alpha:
                    kk.append("True")
                else:
                    kk.append("False")
                ####################################################################################################################
                # perform the Shapiro-Wilk test
                test_statistic, p_value = shapiro(data)

                # interpret the result
                alpha = 0.05
                if p_value > alpha:
                    kk.append("True")
                else:
                    kk.append("False")
                
                ######################################################################################################################
                ######################################################################################################################
                #fileWrite.write("##########################################################")
                #fileWrite.write("#************ Removing numerical data Outliers ***********")
                #fileWrite.write("##########################################################")
                if kk.count("False") >= 2:
                    percentile25=data.quantile(0.25)
                    percentile75=data.quantile(0.75)
                    iqr=percentile75-percentile25
                    upper_limit=percentile75+1.5*iqr
                    lower_limit=percentile25-1.5*iqr
                    uplen=len(df[data>upper_limit])
                    uplen_percent=(uplen/len(data))*100
                    lowlen=len(df[data<lower_limit])
                    lowlen_percent=(lowlen/len(data))*100
                    if int(uplen_percent)==0:
                        pass
                    elif int(uplen_percent)<=5:

                        for lk5 in range(len(data)):
                            if data[lk5]>upper_limit:
                                row_to_drp.append(lk5)
                            else:
                                pass
                    else:
                        for lk6 in range(len(data)):
                            if data[lk6]>upper_limit:
                                df[col_nm].replace(data[lk6],upper_limit,inplace=True)
                            else:
                                pass

                        fileWrite.write(f'for lk6 in range(len(data)):\n')
                        fileWrite.write(f'    if data[lk6]>{upper_limit}:\n')
                        fileWrite.write(f'        df["{col_nm}"].replace(data[lk6],{upper_limit},inplace=True\n')
                        fileWrite.write(f'    else:\n')
                        fileWrite.write(f'        pass\n\n')



                    if int(lowlen_percent)==0:
                        pass

                        
                    elif int(lowlen_percent)<=5:
                        for lk7 in range(len(data)):
                            if data[lk7]<lower_limit:
                                row_to_drp.append(lk7)
                            else:
                                pass
                    else:
                        for lk8 in range(len(data)):
                            if data[lk8]<lower_limit:
                                df[col_nm].replace(data[lk8],lower_limit,inplace=True)

                            else:
                                pass

                        fileWrite.write(f'for lk8 in range(len(data)):\n')
                        fileWrite.write(f'    if data[lk8]<{lower_limit}:\n')
                        fileWrite.write(f'        df["{col_nm}"].replace(data[lk8],{lower_limit},inplace=True)\n')
                        fileWrite.write(f'    else:\n')
                        fileWrite.write(f'        pass\n\n')
                    del percentile25
                    del percentile75
                    del iqr
                    del upper_limit
                    del lower_limit
                    del uplen
                    del uplen_percent
                    del lowlen
                    del lowlen_percent

                else:
                    
                    upper_lim=data.mean()+3*data.std()
                    lower_lim=data.mean()-3*data.std()
                    outliers_up=data>upper_lim
                    outliers_down=data<lower_lim
                    outliers_up_len=len(df[data>outliers_up])
                    outliers_down_len=len(df[data<outliers_down])
                    outliers_up_percent=(outliers_up_len/len(data))*100
                    outliers_down_percent=(outliers_down_len/len(data))*100
                    


                    if int(outliers_up_percent)==0:
                        pass

                    elif int(outliers_up_percent)<=5:
                        for lk in range(len(data)):
                            if data[lk]>upper_lim:
                                row_to_drp.append(lk)
                            else:
                                pass
                    else:
                        for lk2 in range(len(data)):
                            if data[lk2]>upper_lim:
                                
                                df[col_nm].replace(data[lk2],upper_lim,inplace=True)

                            else:
                                pass

                        fileWrite.write(f'for lk2 in range(len(data)):\n')
                        fileWrite.write(f'    if data[lk2]>{upper_lim}:\n')
                        fileWrite.write(f'        df["{col_nm}"].replace(data[lk2],{upper_lim},inplace=True)\n')
                        fileWrite.write(f'    else:\n')
                        fileWrite.write(f'        pass\n\n')

                    if int(outliers_down_percent)==0:
                        pass
                    elif int(outliers_down_percent)<=5:
                        for lk3 in range(len(data)):
                            if data[lk3]<lower_lim:
                                row_to_drp.append(lk3)
                            else:
                                pass
                    else:
                        for lk4 in range(len(data)):
                            if data[lk4]<lower_lim:
                                df[col_nm].replace(data[lk4],lower_lim,inplace=True)

                            else:
                                pass
                        fileWrite.write(f'for lk4 in range(len(data)):\n')
                        fileWrite.write(f'    if data[lk4]<{lower_lim}:\n')
                        fileWrite.write(f'        df["{col_nm}"].replace(data[lk4],{lower_lim},inplace=True)\n')
                        fileWrite.write(f'    else:\n')
                        fileWrite.write(f'        pass\n\n')

                    del upper_lim 
                    del lower_lim
                    del outliers_up
                    del outliers_down
                    del outliers_up_len
                    del outliers_down_len
                    del outliers_up_percent



                kk.clear()
                nn=nn+1
                del data
                del col_nm
                del result
                del alpha
                del test_statistic
                del p_value


            if len(row_to_drp)>0:
                row_drp_rmv_outl = list(set(row_to_drp))
                df=df.drop(row_drp_rmv_outl).reset_index(drop=True)
                fileWrite.write(f'row_drp_rmv_outl={row_drp_rmv_outl}\n')
                fileWrite.write('df=df.drop(row_drp_rmv_outl).reset_index(drop=True)\n\n\n')
            else:
                pass

        except Exception as e:
            pass
        
    else:
        pass
    
    #######################################################
    #********************* Handeling high cardinality ***************
    ########################################################
    if "option3" in additional_options:
        fileWrite.write("#############################################################\n")
        fileWrite.write("#*********** Handeling high cardinality *********************\n")
        fileWrite.write("##############################################################\n\n")
        xp=[]
        for io in list(df.select_dtypes(include="object").columns):
            if df[io].nunique() >=60 and io!=target_col_name:
                xp.append(io)
                
            else:
                pass
            
        for i in xp:
            if drop_un.lower()=="drop":
                df.drop(i, axis=1, inplace=True)
                fileWrite.write(f'df.drop({i}, axis=1, inplace=True)\n\n')
            elif drop_un[0]=='1' or drop_un[0]=='2' or drop_un[0]=='3' or drop_un[0]=='4' or drop_un[0]=='5' or drop_un[0]=='6' or drop_un[0]=='7' or drop_un[0]=='8' or drop_un[0]=='9':
                count_series = df[i].value_counts()
                top_cat = count_series.nlargest(int(drop_un))
                un_val=top_cat.index.tolist()
                df.loc[~df[i].isin(un_val), i] = 'Other'
                fileWrite.write(f'df.loc[~df["{i}"].isin({un_val}), "{i}"] = "Other"\n\n')
            elif drop_un.lower()=="default":
                if df[io].nunique() ==df.shape[0] or  df[io].nunique() >=df.shape[0]-100:
                    df.drop(i,axis-1,inplace=True)
                    fileWrite.write(f'df.drop("{i}",axis-1,inplace=True)\n\n')
                elif tmpDF[io].nunique() >=60:
                    if df.shape[0]/2>=120:
                        count_series = df[i].value_counts()
                        top_sixty = count_series.nlargest(df.shape[0]/2+30)
                        un_val=top_sixty.index.tolist()
                        df.loc[~df[i].isin(un_val), i] = 'Other'
                        fileWrite.write(f'df.loc[~df["{i}"].isin({un_val}), "{i}"] = "Other"\n\n')
                    else:
                        count_series = df[i].value_counts()
                        top_sixty = count_series.nlargest(60)
                        un_val=top_sixty.index.tolist()
                        df.loc[~df[i].isin(un_val), i] = 'Other'
                        fileWrite.write(f'df.loc[~df["{i}"].isin({un_val}), "{i}"] = "Other"\n\n')
                else:
                    pass
            elif drop_un.lower()=="None":
                pass
                
            else:
                pass
                
        fileWrite.write('\n') 
        
    else:
        pass
    
    
    #######################################################
    #******************* EDA *************************
    ########################################################
    if "option4" in additional_options:
        #***********EDA for categorical data********************
        fileWrite.write('##########################################\n')
        fileWrite.write('#******************* EDA ******************\n')
        fileWrite.write('##########################################\n\n\n')
        eda_cat_co=[]
        for i in df.select_dtypes(include="object").columns:
            if df[i].nunique()<=15:
                eda_cat_co.append(i)
            else:
                pass
            
        
        
        #Countplot
        if len(eda_cat_co)>0:
            fileWrite.write('#******************* count plot for all categorical data ******************\n')
        else:
            pass
        nn=1
        for j in range(len(eda_cat_co)):
            
            if len(eda_cat_co)==0:
                pass
            else:
                fileWrite.write(f'plt.subplot({math.ceil(len(eda_cat_co)/3)},3,{nn})\n') 
                fileWrite.write(f'sns.countplot(df["{eda_cat_co[j]}"])\n')
                fileWrite.write('ax=plt.gca()\n')
                fileWrite.write('ax.set_xticklabels(ax.get_xticklabels(), rotation=90)\n')
                fileWrite.write('plt.subplots_adjust(left=0.0,bottom=0.0,right=1.0,top=1.0,wspace=0.8,hspace=0.8)\n\n')
                nn=nn+1
        fileWrite.write('\n')
        del nn
        
        #pie chart
        if len(eda_cat_co)>0:
            fileWrite.write('#******************* pie chart for all categorical data ******************\n')
        else:
            pass
        nn=1
        for j in range(len(eda_cat_co)):
            
            if len(eda_cat_co)==0:
                pass
            else:
                fileWrite.write(f'plt.subplot({math.ceil(len(eda_cat_co)/2)},2,nn)\n')
                fileWrite.write(f'df["{eda_cat_co[j]}"].value_counts().plot(kind="pie",autopct="%1.2f%%",colors=sns.color_palette("bright"),pctdistance=0.8)\n')
                fileWrite.write('plt.subplots_adjust(left=-0.2,bottom=-0.2,right=2.4,top=3.0,wspace=0.0,hspace=0.0)\n\n')
                nn=nn+1
        fileWrite.write('\n')
        del nn
        
        #Heatmap
        if len(eda_cat_co)>0:
            fileWrite.write('#******************* Heatmap categorical data ******************\n')
        else:
            pass

        nnn = 1
        np = []  # Move the initialization of np outside the outer loop

        for j in range(len(eda_cat_co)):
            for k in eda_cat_co:
                if eda_cat_co[j] != k:
                    np.append(k)
                else:
                    pass

            for p in np:
                fileWrite.write(f'plt.subplot({len(eda_cat_co)*5}, 2, {nnn})\n')
                fileWrite.write(f'sns.heatmap(pd.crosstab(df["{eda_cat_co[j]}"], df["{p}"]))\n')
                fileWrite.write(f'plt.subplots_adjust(left=-2.0, bottom=600.0, right=0.0, top=630.0, wspace=0.4, hspace=0.4)\n\n')
                nnn += 1

            np.clear()
        fileWrite.write('\n')
        del nnn
        del np
        
        #***********EDA for numerical data********************
        eda_num_co=df.select_dtypes(include=["int64","float64","int32","float32"]).columns
        
        #Heatmap
        fileWrite.write('#******************* Heatmap ******************\n')
        fileWrite.write('sns.heatmap(df.select_dtypes(include=["int64","float64"]),annot=True,fmt=".1f")\n')
        fileWrite.write('plt.show()\n\n')
        
        #Histogram
        if len(eda_num_co)>0:
            fileWrite.write('#******************* Histogram for all numerical data ******************\n')
        else:
            pass
        nnn=1
        for j in range(len(eda_num_co)):
            
            if len(eda_num_co)==0:
                pass
            else:
                fileWrite.write(f'plt.subplot({math.ceil(len(eda_num_co)/3)},3,nnn)\n')
                fileWrite.write(f'plt.hist(df["{eda_num_co[j]}"])\n')
                fileWrite.write('plt.subplots_adjust(left=0.0,bottom=0.0,right=2.0,top=3.0,wspace=0.4,hspace=0.4)\n\n')
                nnn=nnn+1
        fileWrite.write('\n')
        del nnn
        
        eda_cat_co3=[]
        for i in df.select_dtypes(include="object").columns:
            if df[i].nunique()<=6:
                eda_cat_co3.append(i)
            else:
                pass
        if len(eda_cat_co3)>1:      
            eda_cat_co3.sort()
        else:
            pass

        
        #scatter plot
        if len(eda_num_co)>0:
            fileWrite.write('#******************* scatter plot for all numerical data ******************\n')
        else:
            pass
        if len(eda_cat_co3)==0:
            nnn = 1
            np = []  # Move the initialization of np outside the outer loop

            for j in range(len(eda_num_co)):
                for k in eda_num_co:
                    if eda_num_co[j] != k:
                        np.append(k)
                    else:
                        pass

                for p in np:
                    fileWrite.write(f'plt.subplot({len(eda_num_co)*5}, 2, {nnn})\n')
                    fileWrite.write(f'sns.scatterplot(df["{eda_num_co[j]}"], df["{p}"])\n')
                    fileWrite.write(f'plt.subplots_adjust(left=-2.0, bottom=600.0, right=0.0, top=630.0, wspace=0.4, hspace=0.4)\n\n')
                    nnn += 1

                np.clear()
            fileWrite.write('\n')
        elif len(eda_cat_co3)==1:
            nnn = 1
            np = []  # Move the initialization of np outside the outer loop

            for j in range(len(eda_num_co)):
                for k in eda_num_co:
                    if eda_num_co[j] != k:
                        np.append(k)
                    else:
                        pass

                for p in np:
                    fileWrite.write(f'plt.subplot({len(eda_num_co)*5}, 2, {nnn})\n')
                    fileWrite.write(f'sns.scatterplot(df["{eda_num_co[j]}"], df["{p}"],hue=df["{eda_cat_co3[0]}"])\n')
                    fileWrite.write(f'plt.subplots_adjust(left=-2.0, bottom=600.0, right=0.0, top=630.0, wspace=0.4, hspace=0.4)\n\n')
                    nnn += 1

                np.clear()
            fileWrite.write('\n')
        elif len(eda_cat_co3)==2:
            nnn = 1
            np = []  # Move the initialization of np outside the outer loop

            for j in range(len(eda_num_co)):
                for k in eda_num_co:
                    if eda_num_co[j] != k:
                        np.append(k)
                    else:
                        pass

                for p in np:
                    fileWrite.write(f'plt.subplot({len(eda_num_co)*5}, 2, {nnn})\n')
                    fileWrite.write(f'sns.scatterplot(df["{eda_num_co[j]}"], df["{p}"],hue=df["{eda_cat_co3[1]}"],style=df["{eda_cat_co3[0]}"])\n')
                    fileWrite.write(f'plt.subplots_adjust(left=-2.0, bottom=600.0, right=0.0, top=630.0, wspace=0.4, hspace=0.4)\n\n')
                    nnn += 1

                np.clear()
            fileWrite.write('\n')
        elif len(eda_cat_co3)>3:
            nnn = 1
            np = []  # Move the initialization of np outside the outer loop

            for j in range(len(eda_num_co)):
                for k in eda_num_co:
                    if eda_num_co[j] != k:
                        np.append(k)
                    else:
                        pass

                for p in np:
                    fileWrite.write(f'plt.subplot({len(eda_num_co)*5}, 2, {nnn})\n')
                    fileWrite.write(f'sns.scatterplot(df["{eda_num_co[j]}"], df["{p}"],hue=df["{eda_cat_co3[2]}"],style=df["{eda_cat_co3[1]}"],size=df["{eda_cat_co3[0]}"])\n')
                    fileWrite.write(f'plt.subplots_adjust(left=-2.0, bottom=600.0, right=0.0, top=630.0, wspace=0.4, hspace=0.4)\n\n')
                    nnn += 1

                np.clear()
            fileWrite.write('\n')
        else:
            pass
        del nnn
        del np
        
        #barplot
        eda_cat_co2=[]
        for i in tmpDF.select_dtypes(include="object").columns:
            if tmpDF[i].nunique()<=15:
                eda_cat_co2.append(i)
            else:
                pass
            

        eda_cat_co2=[]
        for i in tmpDF.select_dtypes(include="object").columns:
            if tmpDF[i].nunique()<=15:
                eda_cat_co2.append(i)
            else:
                pass
            

        if len(eda_num_co)>0:
            fileWrite.write('#******************* scatter plot for all numerical data ******************\n')
        else:
            pass
        if len(eda_cat_co3)==0:
            nnn = 1
            np = eda_num_co=tmpDF.select_dtypes(include=["int64","float64"]).columns
            for j in range(len(eda_cat_co2)):


                for p in np:
                    fileWrite.write(f'plt.subplot({len(eda_cat_co2)*5}, 2, {nnn})\n')
                    fileWrite.write(f'sns.barplot(df["{eda_cat_co2[j]}"], df["{p}"])\n')
                    fileWrite.write(f'plt.subplots_adjust(left=-2.0, bottom=600.0, right=0.0, top=630.0, wspace=0.4, hspace=0.4)\n\n')
                    nnn += 1
            fileWrite.write('\n')
        elif len(eda_cat_co3)>=1:
            xp=0
            for lo in eda_cat_co3:
                if xp==4:
                    break
                else:
                    nnn = 1
                    np = eda_num_co=tmpDF.select_dtypes(include=["int64","float64"]).columns
                    for j in range(len(eda_cat_co2)):


                        for p in np:
                            fileWrite.write(f'plt.subplot({len(eda_cat_co2)*5}, 2, {nnn})\n')
                            fileWrite.write(f'sns.barplot(df["{eda_cat_co2[j]}"], df["{p}"],hue=df["{lo}"])\n')
                            fileWrite.write(f'plt.subplots_adjust(left=-2.0, bottom=600.0, right=0.0, top=630.0, wspace=0.4, hspace=0.4)\n\n')
                            nnn += 1
                    fileWrite.write('\n')
        else:
            pass
        
    else:
        pass
    

    #######################################################
    #******************* Encoding *************************
    ########################################################
    
    
    allCatCol=list(df.select_dtypes(include="object").columns)
    if len(allCatCol)==0:
        numerical_column_df=df.select_dtypes(include=["int64","float64","int32","float32"])
        fileWrite.write('numerical_column_df=df.select_dtypes(include=["int64","float64","int32","float32"])\n')
        final_dataset=numerical_column_df
        fileWrite.write('final_dataset=numerical_column_df\n\n\n')
    else:
        fileWrite.write("#######################################################\n")
        fileWrite.write("#*********************** Encoding ********************\n")
        fileWrite.write("#######################################################\n\n")
        fileWrite.write("from sklearn.preprocessing import OneHotEncoder\n")
        fileWrite.write("from sklearn.preprocessing import OrdinalEncoder\n\n")    
        fileWrite.write('allCatCol=list(df.select_dtypes(include="object").columns)\n\n')
        
        #If target column is categorical then appending in rankcol for classification problem
        if target_col_name in allCatCol:
            if rank_col[0].lower()=="none":
                rank_col.clear()
                rank_col.append(target_col_name)
                fileWrite.write('rank_col.clear()\n')
                fileWrite.write('rank_col.append(target_col_name)\n\n')
            else:
                rank_col.append(target_col_name)
                fileWrite.write('rank_col.append(target_col_name)\n\n')
        else:
            pass

        categorical_column=None
        fileWrite.write("categorical_column=None\n\n")

        if rank_col[0]=="none" or rank_col[0]=="None":
            categorical_column=allCatCol
            fileWrite.write(f'categorical_column={allCatCol}\n')
        #below elif condition if the rank col is drop because of contain nan 
        elif all(item in df.columns for item in rank_col)==False:
            categorical_column=allCatCol
            fileWrite.write(f'categorical_column={allCatCol}\n')
        else:
            fileWrite.write("#********** Performing ordinal encoding *************")
            tmp_col_df=df.select_dtypes(include="object")
            fileWrite.write('tmp_col_df=df.select_dtypes(include="object")\n')
            categorical_column_1=tmp_col_df.drop(rank_col,axis=1)
            fileWrite.write(f'categorical_column_1=tmp_col_df.drop({rank_col},axis=1)\n')
            categorical_column=list(categorical_column_1.columns)
            fileWrite.write(f'categorical_column=list(categorical_column_1.columns)\n')
            categorical_column_2=tmp_col_df[rank_col]
            fileWrite.write(f'categorical_column_2=tmp_col_df[rank_col]\n')
            ordinal = OrdinalEncoder()
            fileWrite.write('ordinal = OrdinalEncoder()\n')
            categorical_column_2[rank_col]=ordinal.fit_transform(categorical_column_2)
            fileWrite.write(f'categorical_column_2[rank_col]=ordinal.fit_transform(categorical_column_2)\n')
            df.drop(df[rank_col], axis=1,inplace=True)
            fileWrite.write(f'df.drop(df[rank_col], axis=1,inplace=True)\n')
            df=pd.merge(df,categorical_column_2,left_index=True,right_index=True)
            fileWrite.write('df=pd.merge(df,categorical_column_2,left_index=True,right_index=True)\n\n')
            


        n=0
        nl=[]
        for i in range(len(categorical_column)):
            v=list(df[categorical_column[n]].unique())
            if len(v)==3:
                eno_dt={v[0]:0,v[1]:1,v[2]:2}
                df[categorical_column[n]]=df[categorical_column[n]].map(eno_dt)
                fileWrite.write(f'df["{categorical_column[n]}"]=df["{categorical_column[n]}"].map({eno_dt})\n')
                del eno_dt

            elif len(v)==2:
                eno_dt={v[0]:0,v[1]:1}
                df[categorical_column[n]]=df[categorical_column[n]].map(eno_dt)
                fileWrite.write(f'df["{categorical_column[n]}"]=df["{categorical_column[n]}"].map({eno_dt})\n')
                del eno_dt
            else:
                nl.append(categorical_column[n])

            n=n+1
            del v
            fileWrite.write("\n")


        ll=[]
        encoding= OneHotEncoder(sparse=False)
        fileWrite.write('encoding= OneHotEncoder(sparse=False, drop="first")\n')
        if len(nl)>=1:
            fileWrite.write("#*********** Performing one-Hot encoding ************\n")
            result_encod=encoding.fit_transform(df[nl])
            fileWrite.write(f'result_encod=encoding.fit_transform(df[{nl}])\n')
            dd=pd.get_dummies(df[nl],drop_first=True).keys()
            for i in dd:
                ll.append(i)
            dataFrame_encode_col=pd.DataFrame(result_encod,columns=ll)
            fileWrite.write(f'dataFrame_encode_col=pd.DataFrame(result_encod,columns={ll})\n')
            numerical_column=df.select_dtypes(include=["int64","float64","int32","float32"]).columns
            npl=[]
            for i in numerical_column:
                npl.append(i)
            numeric_col=df[npl]
            
            fileWrite.write(f'numeric_col=df[{npl}]\n')
            #Create dataset
            final_dataset=pd.merge(dataFrame_encode_col,numeric_col,left_index=True,right_index=True)
            fileWrite.write('final_dataset=pd.merge(dataFrame_encode_col,numeric_col,left_index=True,right_index=True)\n\n\n')
            del result_encod
            del dd
            del dataFrame_encode_col
            del numerical_column
            #del npl
        else:
            numerical_column_df=df.select_dtypes(include=["int64","float64","int32","float32"])
            fileWrite.write('numerical_column_df=df.select_dtypes(include=["int64","float64","int32","float32"])\n')
            final_dataset=numerical_column_df
            fileWrite.write('final_dataset=numerical_column_df\n\n\n')
            del numerical_column_df 
        del categorical_column
        del n
        del nl
        del ll
        del encoding
    del allCatCol

    
    #######################################################
    #*** Separating dependent and independent feature *****
    ########################################################
    fileWrite.write('#######################################################\n')
    fileWrite.write('#**** Separating dependent and independent feature ****\n')
    fileWrite.write('#######################################################\n\n')
    x=final_dataset.drop(target_col_name, axis=1)
    y=final_dataset[target_col_name]
        
    fileWrite.write("x=final_dataset.drop(target_col_name, axis=1)\n")
    fileWrite.write("y=final_dataset[target_col_name]\n\n\n")
    

    
    
    #######################################################
    #******** Dropping low variance features  **************
    ########################################################

    if "option5" in additional_options:
        try:
            if x.shape[0]>=1500:
                # compute the variance of each feature

                variances = np.var(x, axis=0).tolist()

                # Sort the list of variances in ascending order
                variances_sorted = sorted(variances)


                v=[]
                # Find the index where the values suddenly increase
                for i in range(1, len(variances_sorted)):
                    if variances_sorted[i] > variances_sorted[i-1]:

                        v.append(format(variances_sorted[i],'.20f'))
                        break
                    else:

                        pass

                


                # instantiate the VarianceThreshold class with the threshold value
                selector = VarianceThreshold(threshold=float(v[0]))

                # fit the selector to the dataframe
                selector.fit(x)

                # get the indices of the low variance features
                low_variance_idx = selector.get_support(indices=True)

                # get the names of the low variance features
                low_variance_features = [col for col in x.columns
                                        if col not in x.columns[low_variance_idx]]
                
                # drop the low variance features from the dataframe

                x.drop(columns=low_variance_features,inplace=True)
                
                if len(low_variance_features)>0:
                    fileWrite.write("##########################################################\n")
                    fileWrite.write("#************ Removing low variance features *************\n")
                    fileWrite.write("###########################################################\n\n")
                    fileWrite.write("from sklearn.feature_selection import VarianceThreshold\n\n")
                    fileWrite.write(f"Threshold_value={v[0]}\n")
                    fileWrite.write(f'selector = VarianceThreshold(threshold=float({v[0]}))\n')
                    fileWrite.write('selector.fit(x)\n')
                    fileWrite.write("low_variance_idx = selector.get_support(indices=True)\n")
                    fileWrite.write("low_variance_features = [col for col in x.columns if col not in x.columns[low_variance_idx]]\n")
                    fileWrite.write(f"x.drop(columns=low_variance_features,inplace=True)\n\n\n")

                else:
                    pass
                del variances 
                del v
                del variances_sorted
                del selector
                del low_variance_idx
                del low_variance_features
            else:
                pass
        except Exception as e:
            pass
       
    else:
        pass
    
    
    ################################################################################
    ######################### Performing VIF for multicollinearity #################
    ################################################################################
    if "option6" in additional_options and request.form['operation'] =="regression":
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # Calculate the VIF for each feature in the training data
            vif_x = pd.DataFrame()
            vif_x["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
            vif_x["features"] = x.columns

            # Print the VIF scores for the training data
            

            # Identify the features with high VIF scores in the training data
            high_vif_train = vif_x[vif_x['VIF Factor'] > 10]['features']
            
            # Drop the highly correlated features from both the training and testing data
            x = x.drop(high_vif_train, axis=1)


            del vif_x
            del high_vif_train

        except Exception as e:
            pass
        
        
        #*************Writing code for VIF multicollinearity**************
        fileWrite.write("#######################################################\n")
        fileWrite.write("#****** Performing VIF for multicollinearity **********\n")
        fileWrite.write("#######################################################\n\n")
        fileWrite.write(f'from statsmodels.stats.outliers_influence import variance_inflation_factor\n\n')

        fileWrite.write(f'#** Calculate the VIF for each feature in the training data **\n')
        fileWrite.write(f'vif_x = pd.DataFrame()\n')
        fileWrite.write(f'vif_x["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]\n')
        fileWrite.write(f'vif_x["features"] = x.columns\n\n')

        fileWrite.write(f'#**** Print the VIF scores for the training data ****\n')
        fileWrite.write(f'print(vif_x)\n\n')

        fileWrite.write(f'#** Identify the features with high VIF scores in the training data **\n')
        fileWrite.write(f'high_vif_train = vif_x[vif_x["VIF Factor"] > 10]["features"]\n')
        fileWrite.write('print(high_vif_train)\n\n')

        fileWrite.write(f'#** Drop the highly correlated features from data **\n')
        fileWrite.write(f'x = x.drop(high_vif_train, axis=1)\n\n\n')
        
    else:
        pass
    
    ################################################################################
    ############## Separating data into train and test¶ ############################
    ################################################################################
    # For non scaling needed algorithm
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1)
    # For scaling needed algorithm
    x_Train,x_Test,y_Train,y_Test=train_test_split(x,y,test_size=0.2, random_state=1)
    
    ################################################################################
    ################################### Scaling ####################################
    ################################################################################
            
    #************ Getting the columns name¶ *******************   
    col_for_scaled=[]
    all_col_nm=x.columns
    for i in range(len(all_col_nm)):
        xx=list(x[all_col_nm[i]].unique())
        if max(xx)>2:
            if all_col_nm[i]==target_col_name:
                pass
            else:
                col_for_scaled.append(all_col_nm[i])
        else:
            pass
        del xx
    del all_col_nm

    col_for_scaled
    
    #*************** scaling the x_Train Data *****************
    if len(col_for_scaled)>0:
        x_Train_cols_for_scale=x_Train[col_for_scaled]
        scaler = StandardScaler()
        scaler.fit(x_Train_cols_for_scale)
        x_Train_transform_dt = scaler.transform(x_Train_cols_for_scale)
        x_Train_scaled_df = pd.DataFrame(x_Train_transform_dt, columns=col_for_scaled)
        x_Train.drop(col_for_scaled,axis=1,inplace=True)
        x_Train.reset_index(drop=True, inplace=True)
        x_Train_scaled_df.reset_index(drop=True, inplace=True)

        x_Train=pd.concat([x_Train, x_Train_scaled_df], axis=1)

        del x_Train_cols_for_scale
        del scaler
        del x_Train_transform_dt
        del x_Train_scaled_df
    else:
        pass

    #******************* Scaling the x_Test data *********************
    if len(col_for_scaled)>0:
        x_Test_cols_for_scale=x_Test[col_for_scaled]
        scaler = StandardScaler()
        scaler.fit(x_Test_cols_for_scale)
        x_Test_transform_dt = scaler.transform(x_Test_cols_for_scale)
        x_Test_scaled_df = pd.DataFrame(x_Test_transform_dt, columns=col_for_scaled)
        x_Test.drop(col_for_scaled,axis=1,inplace=True)
        x_Test.reset_index(drop=True, inplace=True)
        x_Test_scaled_df.reset_index(drop=True, inplace=True)

        x_Test=pd.concat([x_Test, x_Test_scaled_df], axis=1)

        del x_Test_cols_for_scale
        del scaler
        del x_Test_transform_dt
        del x_Test_scaled_df
    else:
        pass
    
    
    
    ################################################################################
    ############################### Separate code ##################################
    ################################################################################
    if request.form['operation'] =="none":
        pass
     
    elif request.form['operation'] =="regression":
        ################################################################################
        ############################## Model training ##################################
        ################################################################################
        sc=[]
        ##############################################################
        ################ Support vector regression ###################
        ##############################################################
        
        svr=supportVectorRegressor() 
        svr.fit(x_Train,y_Train)
        svrtd=svr.predict(x_Test)
        svrr2=r2_score(y_Test,svrtd)
        sc.append(svrr2)
        ##############################################################
        ################# K-nearest neighbors ########################
        ##############################################################
        
        knn=kNeighborsRegressor() 
        knn.fit(x_Train,y_Train)
        knntd=knn.predict(x_Test)
        knnr2=r2_score(y_Test,knntd)
        sc.append(knnr2)
        ##############################################################
        ################# Random Forest Regressor ####################
        ##############################################################
         
        rfr=randomForestRegressor() 
        rfr.fit(x_train,y_train)
        rfrtd=rfr.predict(x_test)
        rfrr2=r2_score(y_test,rfrtd)
        sc.append(rfrr2)
        ##############################################################
        #################### XGBoost Regressor #######################
        ##############################################################
        
        xgb = xGBRegressor() 
        xgb.fit(x_train,y_train)
        xgbtd=xgb.predict(x_test)
        xgbr2=r2_score(y_test,xgbtd)
        sc.append(xgbr2)

        del svr
        del svrtd
        del svrr2
        del knn
        del knntd
        del knnr2
        del rfr
        del rfrtd
        del rfrr2
        del xgb
        del xgbtd
        del xgbr2
        
        
        ################################################################################
        ########## Writting code train, test, scaling ###############
        ################################################################################
        
        if sc.index(max(sc))==0 or  sc.index(max(sc))==1:
            fileWrite.write("############################################################################\n")
            fileWrite.write("#**************** Separating Data into train and test data *****************\n")
            fileWrite.write("############################################################################\n\n")
            fileWrite.write("from sklearn.model_selection import train_test_split\n")
            fileWrite.write("x_Train,x_Test,y_Train,y_Test=train_test_split(x,y,test_size=0.2, random_state=1)\n")
            fileWrite.write('#Change the random_state parameter value. It can change the accuracy\n\n\n')
            if len(col_for_scaled)>0:
                fileWrite.write("#######################################################\n")
                fileWrite.write("#****************** Feature scaling ********************\n")
                fileWrite.write("#######################################################\n\n")
                fileWrite.write("from sklearn.preprocessing import StandardScaler\n\n")

                fileWrite.write("#************ Getting the column name ******************\n")
                fileWrite.write(f"col_for_scaled={col_for_scaled}\n\n")

                fileWrite.write("#************ scaling the x_Train Data *****************\n")
                fileWrite.write("x_Train_cols_for_scale=x_Train[col_for_scaled]\n")
                fileWrite.write("scaler = StandardScaler()\n")
                fileWrite.write("scaler.fit(x_Train_cols_for_scale)\n")
                fileWrite.write("x_Train_transform_dt = scaler.transform(x_Train_cols_for_scale)\n")
                fileWrite.write("x_Train_scaled_df = pd.DataFrame(x_Train_transform_dt, columns=col_for_scaled)\n")
                fileWrite.write("x_Train.drop(col_for_scaled,axis=1,inplace=True)\n")
                fileWrite.write("x_Train.reset_index(drop=True, inplace=True)\n")
                fileWrite.write("x_Train_scaled_df.reset_index(drop=True, inplace=True)\n")
                fileWrite.write("x_Train=pd.concat([x_Train, x_Train_scaled_df], axis=1)\n\n")

                fileWrite.write("#************ scaling the x_Test Data *****************\n")
                fileWrite.write("x_Test_cols_for_scale=x_Test[col_for_scaled]\n")
                fileWrite.write("scaler = StandardScaler()\n")
                fileWrite.write("scaler.fit(x_Test_cols_for_scale)\n")
                fileWrite.write("x_Test_transform_dt = scaler.transform(x_Test_cols_for_scale)\n")
                fileWrite.write("x_Test_scaled_df = pd.DataFrame(x_Test_transform_dt, columns=col_for_scaled)\n")
                fileWrite.write("x_Test.drop(col_for_scaled,axis=1,inplace=True)\n")
                fileWrite.write("x_Test.reset_index(drop=True, inplace=True)\n")
                fileWrite.write("x_Test_scaled_df.reset_index(drop=True, inplace=True)\n")
                fileWrite.write("x_Test=pd.concat([x_Test, x_Test_scaled_df], axis=1)\n\n\n")
            else:
                pass
            
            
            
        elif  sc.index(max(sc))==2 or  sc.index(max(sc))==3:
            fileWrite.write("############################################################################\n")
            fileWrite.write("#**************** Separating Data into train and test data *****************\n")
            fileWrite.write("############################################################################\n\n")
            fileWrite.write("from sklearn.model_selection import train_test_split\n")
            fileWrite.write("x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1)\n")
            fileWrite.write('#Change the random_state parameter value. It can change the accuracy\n\n\n')
            
            
        else:
            pass
    
        ################################################################################
        ########################### Writting code mode training ########################
        ################################################################################
        
        fileWrite.write("############################################################################\n")
        fileWrite.write("#****************************** model training *****************************\n")
        fileWrite.write("############################################################################\n\n")

        fileWrite.write("from sklearn import metrics\n")
        fileWrite.write("from sklearn.metrics import mean_squared_error\n\n")

        if sc.index(max(sc))==0:
            fileWrite.write("from sklearn.svm import SVR\n\n")
            
            fileWrite.write("svr=SVR()\n")
            fileWrite.write("svr.fit(x_Train,y_Train)\n")
            fileWrite.write("svrtd=svr.predict(x_Test)\n")
            fileWrite.write("print('r2 score',r2_score(y_Test,svrtd))\n\n\n")
        elif sc.index(max(sc))==1:
            fileWrite.write("from sklearn.neighbors import KNeighborsRegressor\n\n")
            
            fileWrite.write("knn=KNeighborsRegressor()\n")
            fileWrite.write("knn.fit(x_Train,y_Train)\n")
            fileWrite.write("knntd=knn.predict(x_Test)\n")
            fileWrite.write("print('r2 score',r2_score(y_Test,knntd))\n\n\n")
        elif sc.index(max(sc))==2:
            fileWrite.write("from sklearn.ensemble import RandomForestRegressor\n\n")
            
            fileWrite.write("rfr=RandomForestRegressor()\n")
            fileWrite.write("rfr.fit(x_train,y_train)\n")
            fileWrite.write("rfrtd=rfr.predict(x_test)\n")
            fileWrite.write("print('r2 score',r2_score(y_test,rfrtd))\n\n\n")
        elif sc.index(max(sc))==3:
            fileWrite.write("from xgboost import XGBRegressor\n\n")
            
            fileWrite.write("xgb = XGBRegressor()\n")
            fileWrite.write("xgb.fit(x_train,y_train)\n")
            fileWrite.write("xgbtd=xgb.predict(x_test)\n")
            fileWrite.write("print('r2 score',r2_score(y_test,xgbtd))\n\n\n")
        else:
            pass
        
        ################################################################################
        ################################ Feature selection #############################
        ################################################################################
        
        if "option7" in additional_options:
            fileWrite.write("############################################################################\n")
            fileWrite.write("#****************************** Feature selection *************************\n")
            fileWrite.write("############################################################################\n\n")
            fileWrite.write('from sklearn.metrics import mean_squared_error\n')
            fileWrite.write('from sklearn.feature_selection import RFE\n\n')

            
            piow=[]
            if  x.shape[1]<1000:
                piow.append("none")
                fileWrite.write('#Amount of data is very less. No need to perform feature selection\n\n\n')
            else:
                tnc=x.shape
                if tnc[1]>20:
                    n_feature=[round((80/100) * tnc[1]),round((85/100) * tnc[1]),round((90/100) * tnc[1]),round((95/100) * tnc[1])]
                else:
                    n_feature=tnc 
                feature_list=[]
                select_fea_accr=[]
                if sc.index(max(sc))==0:
                    
                    for i in range(len(n_feature)):

                        svr=supportVectorRegressor(kernel="linear") 
                        svr_feature = RFE(svr, n_features_to_select=n_feature[i])
                        svr_feature.fit(x_Train, y_Train)
                        selected_features = x_Train.columns[svr_feature.support_]
                        feature_list.append(selected_features)
                        X_train_selected = x_Train[selected_features]
                        X_test_selected = x_Test[selected_features]

                        svr.fit(X_train_selected,y_Train)
                        svrtd=svr.predict(X_test_selected)
                        svrr2=r2_score(y_Test,svrtd)
                        select_fea_accr.append(svrr2)

                        del svr
                        del svr_feature
                        del selected_features
                        del X_train_selected
                        del X_test_selected
                        del svrtd
                        del svrr2
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        x_Train=x_Train[feature_list[0]]
                        x_Test= x_Test[feature_list[0]]
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        x_Train=x_Train[feature_list[1]]
                        x_Test= x_Test[feature_list[1]]
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        x_Train=x_Train[feature_list[2]]
                        x_Test= x_Test[feature_list[2]]
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        x_Train=x_Train[feature_list[3]]
                        x_Test= x_Test[feature_list[3]]
                    else:
                        pass



                elif sc.index(max(sc))==1:
                    
                    for i in range(len(n_feature)):
                        
                        rfr = randomForestRegressor()
                        knn=kNeighborsRegressor()
                        knn_feature = RFE(rfr, n_features_to_select=n_feature[i])
                        knn_feature.fit(x_Train, y_Train)
                        selected_features = x_Train.columns[knn_feature.support_]
                        feature_list.append(selected_features)
                        X_train_selected = x_Train[selected_features]
                        X_test_selected = x_Test[selected_features]

                        knn.fit(X_train_selected,y_Train)
                        knntd=knn.predict(X_test_selected)
                        knnr2=r2_score(y_Test,knntd)
                        select_fea_accr.append(knnr2)

                        del rfr
                        del knn_feature
                        del selected_features
                        del X_train_selected
                        del X_test_selected
                        del knntd
                        del knnr2
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        x_Train=x_Train[feature_list[0]]
                        x_Test= x_Test[feature_list[0]]
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        x_Train=x_Train[feature_list[1]]
                        x_Test= x_Test[feature_list[1]]
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        x_Train=x_Train[feature_list[2]]
                        x_Test= x_Test[feature_list[2]]
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        x_Train=x_Train[feature_list[3]]
                        x_Test= x_Test[feature_list[3]]
                    else:
                        pass

                elif sc.index(max(sc))==2:
                    
                    for i in range(len(n_feature)):

                        rfr=randomForestRegressor()
                        rfr_feature = RFE(rfr, n_features_to_select=n_feature[i])
                        rfr_feature.fit(x_train, y_train)
                        selected_features = x_train.columns[rfr_feature.support_]
                        feature_list.append(selected_features)
                        X_train_selected = x_train[selected_features]
                        X_test_selected = x_test[selected_features]


                        rfr.fit(X_train_selected,y_train)
                        rfrtd=rfr.predict(X_test_selected)
                        rfrr2=r2_score(y_Test,rfrtd)
                        select_fea_accr.append(rfrr2)

                        del rfr
                        del rfr_feature
                        del selected_features
                        del X_train_selected
                        del X_test_selected
                        del rfrtd
                        del rfrr2

                    if select_fea_accr.index(max(select_fea_accr))==0:
                        x_train=x_train[feature_list[0]]
                        x_test= x_test[feature_list[0]]
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        x_train=x_train[feature_list[1]]
                        x_test= x_test[feature_list[1]]
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        x_train=x_train[feature_list[2]]
                        x_test= x_test[feature_list[2]]
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        x_train=x_train[feature_list[3]]
                        x_test= x_test[feature_list[3]]
                    else:
                        pass

                elif sc.index(max(sc))==3:
                    
                    for i in range(len(n_feature)):
                        
                        xgb = xGBRegressor()
                        xgb_feature = RFE(xgb, n_features_to_select=n_feature[i])
                        xgb_feature.fit(x_train, y_train)
                        selected_features = x_train.columns[xgb_feature.support_]
                        feature_list.append(selected_features)
                        X_train_selected = x_train[selected_features]
                        X_test_selected = x_test[selected_features]

                        xgb.fit(X_train_selected,y_train)
                        xgbtd=xgb.predict(X_test_selected)
                        xgbr2=r2_score(y_Test,xgbtd)
                        select_fea_accr.append(xgbr2)

                        del xgb
                        del xgb_feature
                        del selected_features
                        del X_train_selected
                        del X_test_selected
                        del xgbtd
                        del xgbr2
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        x_train=x_train[feature_list[0]]
                        x_test= x_test[feature_list[0]]
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        x_train=x_train[feature_list[1]]
                        x_test= x_test[feature_list[1]]
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        x_train=x_train[feature_list[2]]
                        x_test= x_test[feature_list[2]]
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        x_train=x_train[feature_list[3]]
                        x_test= x_test[feature_list[3]]
                    else:
                        pass
                else:
                    pass

            
            ################################################################################
            ######################## Feature selection writting code #######################
            ################################################################################
            
            if len(piow)>0:
                pass
            else:
                if sc.index(max(sc))==0:
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        fileWrite.write('svr=SVR(kernel="linear")\n')
                        fileWrite.write(f'svr_feature = RFE(svr, n_features_to_select={n_feature[0]})\n')
                        fileWrite.write('svr_feature.fit(x_Train, y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[svr_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('svr.fit(x_Train,y_Train)\n')
                        fileWrite.write('svrtd=svr.predict(x_Test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,svrtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        fileWrite.write('svr=SVR(kernel="linear")\n')
                        fileWrite.write(f'svr_feature = RFE(svr, n_features_to_select={n_feature[1]})\n')
                        fileWrite.write('svr_feature.fit(x_Train, y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[svr_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('svr.fit(x_Train,y_Train)\n')
                        fileWrite.write('svrtd=svr.predict(x_Test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,svrtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        fileWrite.write('svr=SVR(kernel="linear")\n')
                        fileWrite.write(f'svr_feature = RFE(svr, n_features_to_select={n_feature[2]})\n')
                        fileWrite.write('svr_feature.fit(x_Train, y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[svr_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('svr.fit(x_Train,y_Train)\n')
                        fileWrite.write('svrtd=svr.predict(x_Test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,svrtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        fileWrite.write('svr=SVR(kernel="linear")\n')
                        fileWrite.write(f'svr_feature = RFE(svr, n_features_to_select={n_feature[3]})\n')
                        fileWrite.write('svr_feature.fit(x_Train, y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[svr_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('svr.fit(x_Train,y_Train)\n')
                        fileWrite.write('svrtd=svr.predict(x_Test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,svrtd))\n\n\n')
                    else:
                        pass



                elif sc.index(max(sc))==1:
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        fileWrite.write('from sklearn.ensemble import RandomForestRegressor\n\n')

                        fileWrite.write('knn=KNeighborsRegressor()\n')
                        fileWrite.write('rfr = RandomForestRegressor()\n')
                        fileWrite.write(f'knn_feature = RFE(rfr, n_features_to_select={n_feature[0]})\n')
                        fileWrite.write('knn_feature.fit(x_Train,y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[knn_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('knn.fit(x_Train,y_Train)\n')
                        fileWrite.write('knntd=knn.predict(x_Test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,knntd))\n\n\n')

                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        fileWrite.write('from sklearn.ensemble import RandomForestRegressor\n\n')

                        fileWrite.write('knn=KNeighborsRegressor()\n')
                        fileWrite.write('rfr = RandomForestRegressor()\n')
                        fileWrite.write(f'knn_feature = RFE(rfr, n_features_to_select={n_feature[1]})\n')
                        fileWrite.write('knn_feature.fit(x_Train,y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[knn_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('knn.fit(x_Train,y_Train)\n')
                        fileWrite.write('knntd=knn.predict(x_Test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,knntd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        fileWrite.write('from sklearn.ensemble import RandomForestRegressor\n\n')

                        fileWrite.write('knn=KNeighborsRegressor()\n')
                        fileWrite.write('rfr = RandomForestRegressor()\n')
                        fileWrite.write(f'knn_feature = RFE(rfr, n_features_to_select={n_feature[2]})\n')
                        fileWrite.write('knn_feature.fit(x_Train,y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[knn_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('knn.fit(x_Train,y_Train)\n')
                        fileWrite.write('knntd=knn.predict(x_Test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,knntd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        fileWrite.write('from sklearn.ensemble import RandomForestRegressor\n\n')

                        fileWrite.write('knn=KNeighborsRegressor()\n')
                        fileWrite.write('rfr = RandomForestRegressor()\n')
                        fileWrite.write(f'knn_feature = RFE(rfr, n_features_to_select={n_feature[3]})\n')
                        fileWrite.write('knn_feature.fit(x_Train,y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[knn_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('knn.fit(x_Train,y_Train)\n')
                        fileWrite.write('knntd=knn.predict(x_Test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,knntd))\n\n\n')
                    else:
                        pass

                elif sc.index(max(sc))==2:
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        fileWrite.write('rfr=RandomForestRegressor()\n')
                        fileWrite.write(f'rfr_feature = RFE(rfr, n_features_to_select={n_feature[0]})\n')
                        fileWrite.write('rfr_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[rfr_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('rfr.fit(x_train,y_train)\n')
                        fileWrite.write('rfrtd=rfr.predict(x_test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,rfrtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        fileWrite.write('rfr=RandomForestRegressor()\n')
                        fileWrite.write(f'rfr_feature = RFE(rfr, n_features_to_select={n_feature[1]})\n')
                        fileWrite.write('rfr_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[rfr_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('rfr.fit(x_train,y_train)\n')
                        fileWrite.write('rfrtd=rfr.predict(x_test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,rfrtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        fileWrite.write('rfr=RandomForestRegressor()\n')
                        fileWrite.write(f'rfr_feature = RFE(rfr, n_features_to_select={n_feature[2]})\n')
                        fileWrite.write('rfr_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[rfr_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('rfr.fit(x_train,y_train)\n')
                        fileWrite.write('rfrtd=rfr.predict(x_test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,rfrtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        fileWrite.write('rfr=RandomForestRegressor()\n')
                        fileWrite.write(f'rfr_feature = RFE(rfr, n_features_to_select={n_feature[3]})\n')
                        fileWrite.write('rfr_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[rfr_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('rfr.fit(x_train,y_train)\n')
                        fileWrite.write('rfrtd=rfr.predict(x_test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,rfrtd))\n\n\n')
                    else:
                        pass

                elif sc.index(max(sc))==3:
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        fileWrite.write('xgb = XGBRegressor()\n')
                        fileWrite.write(f'xgb_feature = RFE(xgb, n_features_to_select={n_feature[0]})\n')
                        fileWrite.write('xgb_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[xgb_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('xgb.fit(x_train,y_train)\n')
                        fileWrite.write('xgbtd=xgb.predict(x_test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,xgbtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        fileWrite.write('xgb = XGBRegressor()\n')
                        fileWrite.write(f'xgb_feature = RFE(xgb, n_features_to_select={n_feature[1]})\n')
                        fileWrite.write('xgb_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[xgb_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('xgb.fit(x_train,y_train)\n')
                        fileWrite.write('xgbtd=xgb.predict(x_test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,xgbtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        fileWrite.write('xgb = XGBRegressor()\n')
                        fileWrite.write(f'xgb_feature = RFE(xgb, n_features_to_select={n_feature[2]})\n')
                        fileWrite.write('xgb_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[xgb_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('xgb.fit(x_train,y_train)\n')
                        fileWrite.write('xgbtd=xgb.predict(x_test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,xgbtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        fileWrite.write('xgb = XGBRegressor()\n')
                        fileWrite.write(f'xgb_feature = RFE(xgb, n_features_to_select={n_feature[3]})\n')
                        fileWrite.write('xgb_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[xgb_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('xgb.fit(x_train,y_train)\n')
                        fileWrite.write('xgbtd=xgb.predict(x_test)\n')
                        fileWrite.write('print("r2 score",r2_score(y_Test,xgbtd))\n\n\n')
                    else:
                        pass
                else:
                    pass
           
        else:
            pass
        
        
        ################################################################################
        ################################# Parameter Tuning #############################
        ################################################################################
        if "option8" in additional_options:
            if sc.index(max(sc))==0:
                
                svrparam=[
                        {'C': [3, 5, 7, 9, 0.1, 1, 10, 100, 1000],
                        "kernel":['rbf','linear','sigmoid'],
                        "gamma":[0.1,1, 0.1, 0.01, 0.001, 0.0001,0,2,0.3,0.4,0.5,0.6,0.7]}
                        ]
                fileWrite.write('svrparam=[{"C": [3, 5, 7, 9, 0.1, 1, 10, 100, 1000],"kernel":["rbf","linear","sigmoid"],"gamma":[0.1,1, 0.1, 0.01, 0.001, 0.0001,0,2,0.3,0.4,0.5,0.6,0.7]}]\n\n')

                #Performing GridSearchCV
                fileWrite.write('#Performing GridSearchCV\n')
                svr=supportVectorRegressor() 
                fileWrite.write('svr=SVR()\n')
                svrGrid_search=GridSearchCV(estimator=svr,param_grid=svrparam, cv=9, n_jobs=-1)
                fileWrite.write('svrGrid_search=GridSearchCV(estimator=svr,param_grid=svrparam, cv=9, n_jobs=-1)\n')
                svrGrid_search.fit(x_Train,y_Train)
                fileWrite.write('svrGrid_search.fit(x_Train,y_Train)\n\n')

                #Getting the best score
                fileWrite.write('#Getting the best score\n')
                svraccuracy=svrGrid_search.best_score_
                fileWrite.write('print("best score",svrGrid_search.best_score_)\n\n')
                
                

                fileWrite.write('#Getting the best parameters\n')
                
                fileWrite.write('print(f"best parameters:{svrGrid_search.best_params_}")\n\n\n')
                
                fileWrite.write("###########################################################\n")
                fileWrite.write("#********** Training Model using best parameters **********\n")
                fileWrite.write("###########################################################\n\n")
                svr = supportVectorRegressor(C=svrGrid_search.best_params_['C'],gamma=svrGrid_search.best_params_['gamma'],kernel=svrGrid_search.best_params_['kernel']) 
                
                fileWrite.write(f"svr = SVR(C=svrGrid_search.best_params_['C'],gamma=svrGrid_search.best_params_['gamma'],kernel=svrGrid_search.best_params_['kernel'])\n")
                svr.fit(x_Train,y_Train)
                fileWrite.write('svr.fit(x_Train,y_Train)\n')
                
                fileWrite.write('print("score:",svr.score(x_Test,y_Test))\n\n\n')
            ##############################################################
            #################### KNN Tuning ##############################
            ##############################################################    
            elif sc.index(max(sc))==1:
                
                knnparam=[
                    {'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'leaf_size': [15, 20]}
                    ]
                fileWrite.write('knnparam=[{"n_neighbors": [3, 5, 7, 9],"weights": ["uniform", "distance"],"leaf_size": [15, 20]}]\n\n')

                #Performing GridSearchCV
                fileWrite.write('#Performing GridSearchCV\n')
                knn=kNeighborsRegressor() 
                fileWrite.write('knn=KNeighborsRegressor()\n')
                knnGrid_search=GridSearchCV(estimator=knn,param_grid=knnparam, cv=9, n_jobs=-1)
                fileWrite.write('knnGrid_search=GridSearchCV(estimator=knn,param_grid=knnparam, cv=9, n_jobs=-1)\n')
                knnGrid_search.fit(x_Train,y_Train)
                fileWrite.write('knnGrid_search.fit(x_Train,y_Train)\n\n')

                #Getting the best score
                fileWrite.write('#Getting the best score\n')
                knnaccuracy=knnGrid_search.best_score_
                fileWrite.write('print("best score",knnGrid_search.best_score_)\n\n')
                
                
                fileWrite.write('#Getting the best parameters\n')
                
                fileWrite.write('print(f"best parameters:{knnGrid_search.best_params_}")\n\n\n')
                
                fileWrite.write("###########################################################\n")
                fileWrite.write("#********** Training Model using best parameters **********\n")
                fileWrite.write("###########################################################\n\n")
                knn = kNeighborsRegressor(leaf_size=knnGrid_search.best_params_['leaf_size'],n_neighbors=knnGrid_search.best_params_['n_neighbors'],weights=knnGrid_search.best_params_['weights']) 
                
                fileWrite.write(f"knn = KNeighborsRegressor(leaf_size=knnGrid_search.best_params_['leaf_size'],n_neighbors=knnGrid_search.best_params_['n_neighbors'],weights=knnGrid_search.best_params_['weights'])\n")
                knn.fit(x_Train,y_Train)
                fileWrite.write('knn.fit(x_Train,y_Train)\n')
                
                fileWrite.write('print("score:",knn.score(x_Test,y_Test))\n\n\n')
            ##############################################################
            #################### RFR Tuning ##############################
            ##############################################################    
            elif sc.index(max(sc))==2:  
                
                rfrparam=[
                    {"n_estimators":[100,150,170,200],
                    "criterion":["squared_error", "absolute_error"],
                    "max_depth":[10, 24, 39, 53, 68, 82, 97, 120],
                    "min_samples_split":[2, 6, 10],"min_samples_leaf":[1, 3, 4],
                    }
                    ]
                fileWrite.write('rfrparam=[{"n_estimators":[1,10,20,70,100],"criterion":["squared_error", "absolute_error"],"max_depth":[10, 24, 39, 53, 68, 82, 97, 120],"min_samples_split":[2, 6, 10],"min_samples_leaf":[1, 3, 4]}]\n\n')

                #Performing GridSearchCV
                fileWrite.write('#Performing GridSearchCV\n')
                rfr=randomForestRegressor() 
                fileWrite.write('rfr=RandomForestRegressor()\n')
                rfrGrid_search=GridSearchCV(estimator=rfr,param_grid=rfrparam, cv=9, n_jobs=-1)
                fileWrite.write('rfrGrid_search=GridSearchCV(estimator=rfr,param_grid=rfrparam, cv=9, n_jobs=-1)\n')
                rfrGrid_search.fit(x_train,y_train)
                fileWrite.write('rfrGrid_search.fit(x_train,y_train)\n\n')

                #Getting the best score
                fileWrite.write('#Getting the best score\n')
                rfraccuracy=rfrGrid_search.best_score_
                fileWrite.write('print("best score",rfrGrid_search.best_score_)\n\n')
                
                

                #Getting the best parameters
                fileWrite.write('#Getting the best parameters\n')
                
                fileWrite.write('print(f"best parameters:{rfrGrid_search.best_params_}")\n\n\n')
                
                fileWrite.write("###########################################################\n")
                fileWrite.write("#********** Training Model using best parameters **********\n")
                fileWrite.write("###########################################################\n\n")
                rfr = randomForestRegressor(n_estimators=rfrGrid_search.best_params_['n_estimators'],criterion=rfrGrid_search.best_params_['criterion'],max_depth=rfrGrid_search.best_params_['max_depth'],
                    min_samples_split=rfrGrid_search.best_params_['min_samples_split'],min_samples_leaf=rfrGrid_search.best_params_['min_samples_leaf'])
                
                fileWrite.write(f"rfr = RandomForestRegressor(n_estimators=rfrGrid_search.best_params_['n_estimators'],criterion=rfrGrid_search.best_params_['criterion'],max_depth=rfrGrid_search.best_params_['max_depth'],min_samples_split=rfrGrid_search.best_params_['min_samples_split'],min_samples_leaf=rfrGrid_search.best_params_['min_samples_leaf']\n")
                rfr.fit(x_train,y_train)
                fileWrite.write('rfr.fit(x_train,y_train)\n')
                
                fileWrite.write('print("score:",rfr.score(x_test,y_test))\n\n\n')
            ##############################################################
            #################### XGBoost Tuning ##########################
            ##############################################################    
            elif sc.index(max(sc))==3:
                
                xgbparam=[
                    
                    { 'min_child_weight': [2, 5, 8],
                    'gamma': [0, 0.1, 0.2 ],
                    'subsample': [0.6, 0.8],
                    'colsample_bytree': [0.6, 0.8],
                        'max_depth': [3,  5, 7, 9],
                        "learning_rate":[ 0.1, 0.01,  0.2,  0.3],
                        "n_estimators":[100,150, 250, 400 ]}
                        ]
                fileWrite.write('xgbparam=[{ "min_child_weight": [2, 5, 8],"gamma": [0, 0.1, 0.2],"subsample": [0.6, 0.8, 1.0],"colsample_bytree": [0.6, 0.8],"max_depth": [3,  5, 7, 9],"learning_rate":[0.1, 0.01,  0.2,  0.3],"n_estimators":[100,150, 250, 400]}]\n\n')

                #Performing GridSearchCV
                fileWrite.write('#Performing GridSearchCV\n')
                xgb = xGBRegressor()
                fileWrite.write('xgb = XGBRegressor()\n')
                xgbGrid_search=GridSearchCV(estimator=xgb,param_grid=xgbparam, cv=9, n_jobs=-1)
                fileWrite.write('xgbGrid_search=GridSearchCV(estimator=xgb,param_grid=xgbparam, cv=9, n_jobs=-1)\n')
                xgbGrid_search.fit(x_train,y_train)
                fileWrite.write('xgbGrid_search.fit(x_train,y_train)\n\n')

                #Getting the best score
                fileWrite.write('#Getting the best score\n')
                xgbaccuracy=xgbGrid_search.best_score_
                fileWrite.write('print("best score",xgbGrid_search.best_score_)\n\n\n')
                
                
                #Getting the best parameters
                fileWrite.write('#Getting the best parameters\n')
                
                fileWrite.write('print(f"best parameters:{xgbGrid_search.best_params_}")\n\n\n')
                
                fileWrite.write("###########################################################\n")
                fileWrite.write("#********** Training Model using best parameters **********\n")
                fileWrite.write("###########################################################\n\n")
                xgb = xGBRegressor(colsample_bytree=rfrGrid_search.best_params_['colsample_bytree'],gamma=rfrGrid_search.best_params_['gamma'],learning_rate=rfrGrid_search.best_params_['learning_rate'],
                    max_depth=rfrGrid_search.best_params_['max_depth'],min_child_weight=rfrGrid_search.best_params_['min_child_weight'],n_estimators=rfrGrid_search.best_params_['n_estimators'],
                    subsample=rfrGrid_search.best_params_['subsample'])

                fileWrite.write(f"xgb = XGBRegressor(colsample_bytree=rfrGrid_search.best_params_['colsample_bytree'],gamma=rfrGrid_search.best_params_['gamma'],learning_rate=rfrGrid_search.best_params_['learning_rate'],max_depth=rfrGrid_search.best_params_['max_depth'],min_child_weight=rfrGrid_search.best_params_['min_child_weight'],n_estimators=rfrGrid_search.best_params_['n_estimators'],subsample=rfrGrid_search.best_params_['subsample'])\n")
                xgb.fit(x_train,y_train)
                fileWrite.write('xgb.fit(x_train,y_train)\n')
                
                fileWrite.write('print("score:",xgb.score(x_test,y_test))\n\n\n')

            else:
                pass
            
        else:
            pass

        
        
        ################################################################################
        ################################# Model evaluation #############################
        ################################################################################
        if "option9" in additional_options:
            fileWrite.write("######################################\n")
            fileWrite.write("#****** Model evaluation ************\n")
            fileWrite.write("######################################\n\n")
            if sc.index(max(sc))==0:
                # make predictions on the testing set
                fileWrite.write('# make predictions on the testing set\n')
                fileWrite.write('y_pred = svr.predict(x_Test)\n\n')

                # calculate the evaluation metrics
                fileWrite.write('# calculate the evaluation metrics\n\n')
                fileWrite.write('#Mean Squared error\n')
                fileWrite.write('mse = mean_squared_error(y_Test, y_pred)\n')
                fileWrite.write('print(mse)\n\n')
                
                fileWrite.write('#Root mean squared error\n')
                fileWrite.write('rmse = np.sqrt(mse)\n')
                fileWrite.write('print(rmse)\n\n')
                
                fileWrite.write('#r2_score\n')
                fileWrite.write('r2 = r2_score(y_Test, y_pred)\n')
                fileWrite.write('print(r2)\n\n')
                
                fileWrite.write('mean absolute error\n')
                fileWrite.write('mae = mean_absolute_error(y_Test, y_pred)\n')
                fileWrite.write('print(mae)\n\n')
                
                fileWrite.write('Mean absolute percentage error\n')
                fileWrite.write('mape = np.mean(np.abs((y_Test - y_pred) / y_Test)) * 100\n')
                fileWrite.write('print(mape)\n\n\n')
                
                fileWrite.write("############################################\n")
                fileWrite.write("#********** Plot the residuals *************\n")
                fileWrite.write("############################################\n\n")
                fileWrite.write('residuals = y_Test - y_pred\n')
                fileWrite.write('plt.scatter(y_pred, residuals)\n')
                fileWrite.write('plt.xlabel("Predicted Values")\n')
                fileWrite.write('plt.ylabel("Residuals")\n')
                fileWrite.write('plt.show()\n\n\n')
            elif sc.index(max(sc))==1:
                # make predictions on the testing set
                fileWrite.write('# make predictions on the testing set\n')
                fileWrite.write('y_pred = knn.predict(x_Test)\n\n')

                # calculate the evaluation metrics
                fileWrite.write('# calculate the evaluation metrics\n\n')
                fileWrite.write('#Mean Squared error\n')
                fileWrite.write('mse = mean_squared_error(y_Test, y_pred)\n')
                fileWrite.write('print(mse)\n\n')
                
                fileWrite.write('#Root mean squared error\n')
                fileWrite.write('rmse = np.sqrt(mse)\n')
                fileWrite.write('print(rmse)\n\n')
                
                fileWrite.write('#r2_score\n')
                fileWrite.write('r2 = r2_score(y_Test, y_pred)\n')
                fileWrite.write('print(r2)\n\n')
                
                fileWrite.write('mean absolute error\n')
                fileWrite.write('mae = mean_absolute_error(y_Test, y_pred)\n')
                fileWrite.write('print(mae)\n\n')
                
                fileWrite.write('Mean absolute percentage error\n')
                fileWrite.write('mape = np.mean(np.abs((y_Test - y_pred) / y_Test)) * 100\n')
                fileWrite.write('print(mape)\n\n\n')
                
                fileWrite.write("############################################\n")
                fileWrite.write("#********** Plot the residuals *************\n")
                fileWrite.write("############################################\n\n")
                fileWrite.write('residuals = y_Test - y_pred\n')
                fileWrite.write('plt.scatter(y_pred, residuals)\n')
                fileWrite.write('plt.xlabel("Predicted Values")\n')
                fileWrite.write('plt.ylabel("Residuals")\n')
                fileWrite.write('plt.show()\n')
            elif sc.index(max(sc))==2:
                # make predictions on the testing set
                fileWrite.write('# make predictions on the testing set\n')
                fileWrite.write('y_pred = rfr.predict(x_test)\n\n')

                # calculate the evaluation metrics
                fileWrite.write('# calculate the evaluation metrics\n\n')
                fileWrite.write('#Mean Squared error\n')
                fileWrite.write('mse = mean_squared_error(y_test, y_pred)\n')
                fileWrite.write('print(mse)\n\n')
                
                fileWrite.write('#Root mean squared error\n')
                fileWrite.write('rmse = np.sqrt(mse)\n')
                fileWrite.write('print(rmse)\n\n')
                
                fileWrite.write('#r2_score\n')
                fileWrite.write('r2 = r2_score(y_test, y_pred)\n')
                fileWrite.write('print(r2)\n\n')
                
                fileWrite.write('mean absolute error\n')
                fileWrite.write('mae = mean_absolute_error(y_test, y_pred)\n')
                fileWrite.write('print(mae)\n\n')
                
                fileWrite.write('Mean absolute percentage error\n')
                fileWrite.write('mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100\n')
                fileWrite.write('print(mape)\n\n\n')
                
                fileWrite.write("############################################\n")
                fileWrite.write("#********** Plot the residuals *************\n")
                fileWrite.write("############################################\n\n")
                fileWrite.write('residuals = y_test - y_pred\n')
                fileWrite.write('plt.scatter(y_pred, residuals)\n')
                fileWrite.write('plt.xlabel("Predicted Values")\n')
                fileWrite.write('plt.ylabel("Residuals")\n')
                fileWrite.write('plt.show()\n\n\n')
            elif sc.index(max(sc))==3:
                # make predictions on the testing set
                fileWrite.write('# make predictions on the testing set\n')
                fileWrite.write('y_pred = xgb.predict(x_test)\n\n')

                # calculate the evaluation metrics
                fileWrite.write('# calculate the evaluation metrics\n\n')
                fileWrite.write('#Mean Squared error\n')
                fileWrite.write('mse = mean_squared_error(y_test, y_pred)\n')
                fileWrite.write('print(mse)\n\n')
                
                fileWrite.write('#Root mean squared error\n')
                fileWrite.write('rmse = np.sqrt(mse)\n')
                fileWrite.write('print(rmse)\n\n')
                
                fileWrite.write('#r2_score\n')
                fileWrite.write('r2 = r2_score(y_test, y_pred)\n')
                fileWrite.write('print(r2)\n\n')
                
                fileWrite.write('mean absolute error\n')
                fileWrite.write('mae = mean_absolute_error(y_test, y_pred)\n')
                fileWrite.write('print(mae)\n\n')
                
                fileWrite.write('Mean absolute percentage error\n')
                fileWrite.write('mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100\n')
                fileWrite.write('print(mape)\n\n\n')
                
                fileWrite.write("############################################\n")
                fileWrite.write("#********** Plot the residuals *************\n")
                fileWrite.write("############################################\n\n")
                fileWrite.write('residuals = y_test - y_pred\n')
                fileWrite.write('plt.scatter(y_pred, residuals)\n')
                fileWrite.write('plt.xlabel("Predicted Values")\n')
                fileWrite.write('plt.ylabel("Residuals")\n')
                fileWrite.write('plt.show()\n\n\n')
            
        else:
            pass    
            
    elif request.form['operation'] == "binary_classification" or request.form['operation'] =="multi_class_classification" :
        
        
        
        ################################################################################
        ################################### Training model  ############################
        ################################################################################
        sc=[]
        ##############################################################
        ################ Support vector classifier ###################
        ##############################################################
        
        svc=supportVectorClassifier()
        svc.fit(x_Train,y_Train)
        svctd=svc.predict(x_Test)
        svcaccuracy = accuracy_score(y_Test, svctd)
        sc.append(svcaccuracy)
        ############################################################
        ########### K-nearest neighbors classifier ##################
        ##############################################################
        
        knn=kNeighborsClassifier()
        knn.fit(x_Train,y_Train)
        knntd=knn.predict(x_Test)
        knnaccuracy = accuracy_score(y_Test, knntd)
        sc.append(knnaccuracy)
        ##############################################################
        ################# Random Forest classifier ####################
        ##############################################################
        
        rfc=randomForestClassifier()
        rfc.fit(x_train,y_train)
        rfctd=rfc.predict(x_test)
        rfcaccuracy = accuracy_score(y_test, rfctd)
        sc.append(rfcaccuracy)
        ##############################################################
        #################### XGBoost classifier #######################
        ##############################################################
        
        xgb = xGBClassifier()
        xgb.fit(x_train,y_train)
        xgbtd=xgb.predict(x_test)
        xgbaccuracy = accuracy_score(y_test, xgbtd)
        sc.append(xgbaccuracy)

        del svc
        del svctd
        del svcaccuracy
        del knn
        del knntd
        del knnaccuracy
        del rfc
        del rfctd
        del rfcaccuracy
        del xgb
        del xgbtd
        del xgbaccuracy
        
        
        
        ################################################################################
        ########## Writting code train, test, scaling ###############
        ################################################################################
        
        if sc.index(max(sc))==0 or  sc.index(max(sc))==1:
            fileWrite.write("############################################################################\n")
            fileWrite.write("#**************** Separating Data into train and test data *****************\n")
            fileWrite.write("############################################################################\n\n")
            fileWrite.write("from sklearn.model_selection import train_test_split\n")
            fileWrite.write("x_Train,x_Test,y_Train,y_Test=train_test_split(x,y,test_size=0.2, random_state=1)\n")
            fileWrite.write('#Change the random_state parameter value. It can change the accuracy\n\n\n')
            if len(col_for_scaled)>0:
                fileWrite.write("#######################################################\n")
                fileWrite.write("#****************** Feature scaling ********************\n")
                fileWrite.write("#######################################################\n\n")
                fileWrite.write("from sklearn.preprocessing import StandardScaler\n\n")

                fileWrite.write("#************ Getting the column name ******************\n")
                fileWrite.write(f"col_for_scaled={col_for_scaled}\n\n")

                fileWrite.write("#************ scaling the x_Train Data *****************\n")
                fileWrite.write("x_Train_cols_for_scale=x_Train[col_for_scaled]\n")
                fileWrite.write("scaler = StandardScaler()\n")
                fileWrite.write("scaler.fit(x_Train_cols_for_scale)\n")
                fileWrite.write("x_Train_transform_dt = scaler.transform(x_Train_cols_for_scale)\n")
                fileWrite.write("x_Train_scaled_df = pd.DataFrame(x_Train_transform_dt, columns=col_for_scaled)\n")
                fileWrite.write("x_Train.drop(col_for_scaled,axis=1,inplace=True)\n")
                fileWrite.write("x_Train.reset_index(drop=True, inplace=True)\n")
                fileWrite.write("x_Train_scaled_df.reset_index(drop=True, inplace=True)\n")
                fileWrite.write("x_Train=pd.concat([x_Train, x_Train_scaled_df], axis=1)\n\n")

                fileWrite.write("#************ scaling the x_Test Data *****************\n")
                fileWrite.write("x_Test_cols_for_scale=x_Test[col_for_scaled]\n")
                fileWrite.write("scaler = StandardScaler()\n")
                fileWrite.write("scaler.fit(x_Test_cols_for_scale)\n")
                fileWrite.write("x_Test_transform_dt = scaler.transform(x_Test_cols_for_scale)\n")
                fileWrite.write("x_Test_scaled_df = pd.DataFrame(x_Test_transform_dt, columns=col_for_scaled)\n")
                fileWrite.write("x_Test.drop(col_for_scaled,axis=1,inplace=True)\n")
                fileWrite.write("x_Test.reset_index(drop=True, inplace=True)\n")
                fileWrite.write("x_Test_scaled_df.reset_index(drop=True, inplace=True)\n")
                fileWrite.write("x_Test=pd.concat([x_Test, x_Test_scaled_df], axis=1)\n\n\n")
            else:
                pass
            
            
            
        elif  sc.index(max(sc))==2 or  sc.index(max(sc))==3:
            fileWrite.write("############################################################################\n")
            fileWrite.write("#**************** Separating Data into train and test data *****************\n")
            fileWrite.write("############################################################################\n\n")
            fileWrite.write("from sklearn.model_selection import train_test_split\n")
            fileWrite.write("x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1)\n")
            fileWrite.write('#Change the random_state parameter value. It can change the accuracy\n\n\n')
            
            
        else:
            pass
    
        ################################################################################
        ########################### Writting code mode training ########################
        ################################################################################
        
        fileWrite.write("############################################################################\n")
        fileWrite.write("#****************************** model training *****************************\n")
        fileWrite.write("############################################################################\n\n")

        fileWrite.write("from sklearn.metrics import accuracy_score\n")
        

        if sc.index(max(sc))==0:
            fileWrite.write("from sklearn.svm import SVC\n\n")
            
            fileWrite.write("svc=SVC()\n")
            fileWrite.write("svc.fit(x_Train,y_Train)\n")
            fileWrite.write("svctd=svc.predict(x_Test)\n")
            fileWrite.write("print('Accuracy',accuracy_score(y_Test, svctd))\n\n\n")
        elif sc.index(max(sc))==1:
            fileWrite.write("from sklearn.neighbors import KNeighborsClassifier\n\n")
            
            fileWrite.write("knn=KNeighborsClassifier()\n")
            fileWrite.write("knn.fit(x_Train,y_Train)\n")
            fileWrite.write("knntd=knn.predict(x_Test)\n")
            fileWrite.write("print('Accuracy',accuracy_score(y_Test, knntd))\n\n\n")
        elif sc.index(max(sc))==2:
            fileWrite.write("from sklearn.ensemble import RandomForestClassifier\n\n")
            
            fileWrite.write("rfc=RandomForestClassifier()\n")
            fileWrite.write("rfc.fit(x_train,y_train)\n")
            fileWrite.write("rfctd=rfc.predict(x_test)\n")
            fileWrite.write("print('Accuracy',accuracy_score(y_test, rfctd))\n\n\n")
        elif sc.index(max(sc))==3:
            fileWrite.write("from xgboost import XGBClassifier\n\n")
            
            fileWrite.write("xgb = XGBClassifier()\n")
            fileWrite.write("xgb.fit(x_train,y_train)\n")
            fileWrite.write("xgbtd=xgb.predict(x_test)\n")
            fileWrite.write("print('Accuracy',accuracy_score(y_test, xgbtd))\n\n\n")
        else:
            pass
            
        ################################################################################
        ################################ Feature selection #############################
        ################################################################################
        
        if "option7" in additional_options:
            fileWrite.write("############################################################################\n")
            fileWrite.write("#****************************** Feature selection *************************\n")
            fileWrite.write("############################################################################\n\n")
            fileWrite.write('from sklearn.feature_selection import RFE\n\n')

            
            piow=[]
            if  x.shape[1]<1000:
                piow.append("none")
                fileWrite.write('#Amount of data is very less. No need to perform feature selection\n\n\n')
            else:
                tnc=x.shape
                if tnc[1]>20:
                    n_feature=[round((80/100) * tnc[1]),round((85/100) * tnc[1]),round((90/100) * tnc[1]),round((95/100) * tnc[1])]
                else:
                    n_feature=tnc 
                feature_list=[]
                select_fea_accr=[]
                if sc.index(max(sc))==0:
                    
                    for i in range(len(n_feature)):

                        svc=supportVectorClassifier(kernel="linear") 
                        svc_feature = RFE(svc, n_features_to_select=n_feature[i])
                        svc_feature.fit(x_Train, y_Train)
                        selected_features = x_Train.columns[svc_feature.support_]
                        feature_list.append(selected_features)
                        X_train_selected = x_Train[selected_features]
                        X_test_selected = x_Test[selected_features]

                        svc.fit(X_train_selected,y_Train)
                        svctd=svc.predict(X_test_selected)
                        svcaccuracy = accuracy_score(y_Test, svctd)
                        select_fea_accr.append(svcaccuracy)
                        

                        del svc
                        del svc_feature
                        del selected_features
                        del X_train_selected
                        del X_test_selected
                        del svctd
                        del svcaccuracy
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        x_Train=x_Train[feature_list[0]]
                        x_Test= x_Test[feature_list[0]]
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        x_Train=x_Train[feature_list[1]]
                        x_Test= x_Test[feature_list[1]]
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        x_Train=x_Train[feature_list[2]]
                        x_Test= x_Test[feature_list[2]]
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        x_Train=x_Train[feature_list[3]]
                        x_Test= x_Test[feature_list[3]]
                    else:
                        pass



                elif sc.index(max(sc))==1:
                    
                    for i in range(len(n_feature)):
                        
                        rfc = randomForestClassifier()
                        knn=kNeighborsClassifier()
                        knn_feature = RFE(rfc, n_features_to_select=n_feature[i])
                        knn_feature.fit(x_Train, y_Train)
                        selected_features = x_Train.columns[knn_feature.support_]
                        feature_list.append(selected_features)
                        X_train_selected = x_Train[selected_features]
                        X_test_selected = x_Test[selected_features]

                        knn.fit(X_train_selected,y_Train)
                        knntd=knn.predict(X_test_selected)
                        knnaccuracy = accuracy_score(y_Test, knntd)
                        select_fea_accr.append(knnaccuracy)

                        del rfr
                        del knn_feature
                        del selected_features
                        del X_train_selected
                        del X_test_selected
                        del knntd
                        del knnaccuracy
                       
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        x_Train=x_Train[feature_list[0]]
                        x_Test= x_Test[feature_list[0]]
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        x_Train=x_Train[feature_list[1]]
                        x_Test= x_Test[feature_list[1]]
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        x_Train=x_Train[feature_list[2]]
                        x_Test= x_Test[feature_list[2]]
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        x_Train=x_Train[feature_list[3]]
                        x_Test= x_Test[feature_list[3]]
                    else:
                        pass

                elif sc.index(max(sc))==2:
                    
                    for i in range(len(n_feature)):

                        rfc=randomForestRegressor()
                        rfc_feature = RFE(rfc, n_features_to_select=n_feature[i])
                        rfc_feature.fit(x_train, y_train)
                        selected_features = x_train.columns[rfc_feature.support_]
                        feature_list.append(selected_features)
                        X_train_selected = x_train[selected_features]
                        X_test_selected = x_test[selected_features]


                        rfc.fit(X_train_selected,y_train)
                        rfctd=rfc.predict(X_test_selected)
                        rfcaccuracy = accuracy_score(y_test, rfctd)
                        select_fea_accr.append(rfcaccuracy)
                        
                        del rfc
                        del rfc_feature
                        del selected_features
                        del X_train_selected
                        del X_test_selected
                        del rfctd
                        del rfcaccuracy
                        

                    if select_fea_accr.index(max(select_fea_accr))==0:
                        x_train=x_train[feature_list[0]]
                        x_test= x_test[feature_list[0]]
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        x_train=x_train[feature_list[1]]
                        x_test= x_test[feature_list[1]]
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        x_train=x_train[feature_list[2]]
                        x_test= x_test[feature_list[2]]
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        x_train=x_train[feature_list[3]]
                        x_test= x_test[feature_list[3]]
                    else:
                        pass

                elif sc.index(max(sc))==3:
                    
                    for i in range(len(n_feature)):
                        
                        xgb = xGBRegressor()
                        xgb_feature = RFE(xgb, n_features_to_select=n_feature[i])
                        xgb_feature.fit(x_train, y_train)
                        selected_features = x_train.columns[xgb_feature.support_]
                        feature_list.append(selected_features)
                        X_train_selected = x_train[selected_features]
                        X_test_selected = x_test[selected_features]

                        xgb.fit(X_train_selected,y_train)
                        xgbtd=xgb.predict(X_test_selected)
                        xgbaccuracy = accuracy_score(y_test, xgbtd)
                        select_fea_accr.append(xgbaccuracy)

                        del xgb
                        del xgb_feature
                        del selected_features
                        del X_train_selected
                        del X_test_selected
                        del xgbtd
                        del xgbaccuracy
                       
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        x_train=x_train[feature_list[0]]
                        x_test= x_test[feature_list[0]]
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        x_train=x_train[feature_list[1]]
                        x_test= x_test[feature_list[1]]
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        x_train=x_train[feature_list[2]]
                        x_test= x_test[feature_list[2]]
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        x_train=x_train[feature_list[3]]
                        x_test= x_test[feature_list[3]]
                    else:
                        pass
                else:
                    pass

            
            ################################################################################
            ######################## Feature selection writting code #######################
            ################################################################################
            
            if len(piow)>0:
                pass
            else:
                if sc.index(max(sc))==0:
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        fileWrite.write('svc=SVC(kernel="linear")\n')
                        fileWrite.write(f'svc_feature = RFE(svc, n_features_to_select={n_feature[0]})\n')
                        fileWrite.write('svc_feature.fit(x_Train, y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[svc_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('svc.fit(x_Train,y_Train)\n')
                        fileWrite.write('svctd=svc.predict(x_Test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, svctd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        fileWrite.write('svc=SVC(kernel="linear")\n')
                        fileWrite.write(f'svc_feature = RFE(svc, n_features_to_select={n_feature[1]})\n')
                        fileWrite.write('svc_feature.fit(x_Train, y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[svc_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('svc.fit(x_Train,y_Train)\n')
                        fileWrite.write('svctd=svc.predict(x_Test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, svctd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        fileWrite.write('svc=SVC(kernel="linear")\n')
                        fileWrite.write(f'svc_feature = RFE(svc, n_features_to_select={n_feature[2]})\n')
                        fileWrite.write('svc_feature.fit(x_Train, y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[svc_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('svc.fit(x_Train,y_Train)\n')
                        fileWrite.write('svctd=svc.predict(x_Test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, svctd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        fileWrite.write('svc=SVC(kernel="linear")\n')
                        fileWrite.write(f'svc_feature = RFE(svc, n_features_to_select={n_feature[3]})\n')
                        fileWrite.write('svc_feature.fit(x_Train, y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[svc_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('svc.fit(x_Train,y_Train)\n')
                        fileWrite.write('svctd=svc.predict(x_Test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, svctd))\n\n\n')
                    else:
                        pass



                elif sc.index(max(sc))==1:
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        fileWrite.write('from sklearn.ensemble import RandomForestClassifier\n\n')
                        
                        fileWrite.write('knn=KNeighborsClassifier()\n')
                        fileWrite.write('rfc = RandomForestClassifier()\n')
                        fileWrite.write(f'knn_feature = RFE(rfc, n_features_to_select={n_feature[0]})\n')
                        fileWrite.write('knn_feature.fit(x_Train,y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[knn_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('knn.fit(x_Train,y_Train)\n')
                        fileWrite.write('knntd=knn.predict(x_Test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, knntd))\n\n\n')

                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        fileWrite.write('from sklearn.ensemble import RandomForestClassifier\n\n')

                        fileWrite.write('knn=KNeighborsClassifier()\n')
                        fileWrite.write('rfc = RandomForestClassifier()\n')
                        fileWrite.write(f'knn_feature = RFE(rfc, n_features_to_select={n_feature[1]})\n')
                        fileWrite.write('knn_feature.fit(x_Train,y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[knn_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('knn.fit(x_Train,y_Train)\n')
                        fileWrite.write('knntd=knn.predict(x_Test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, knntd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        fileWrite.write('from sklearn.ensemble import RandomForestClassifier\n\n')

                        fileWrite.write('knn=KNeighborsClassifier()\n')
                        fileWrite.write('rfc = RandomForestClassifier()\n')
                        fileWrite.write(f'knn_feature = RFE(rfc, n_features_to_select={n_feature[2]})\n')
                        fileWrite.write('knn_feature.fit(x_Train,y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[knn_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('knn.fit(x_Train,y_Train)\n')
                        fileWrite.write('knntd=knn.predict(x_Test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, knntd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        fileWrite.write('from sklearn.ensemble import RandomForestClassifier\n\n')

                        fileWrite.write('knn=KNeighborsClassifier()\n')
                        fileWrite.write('rfc = RandomForestClassifier()\n')
                        fileWrite.write(f'knn_feature = RFE(rfc, n_features_to_select={n_feature[3]})\n')
                        fileWrite.write('knn_feature.fit(x_Train,y_Train)\n')
                        fileWrite.write('selected_features = x_Train.columns[knn_feature.support_]\n')
                        fileWrite.write('x_Train = x_Train[selected_features]\n')
                        fileWrite.write('x_Test = x_Test[selected_features]\n\n')


                        fileWrite.write('knn.fit(x_Train,y_Train)\n')
                        fileWrite.write('knntd=knn.predict(x_Test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, knntd))\n\n\n')
                    else:
                        pass

                elif sc.index(max(sc))==2:
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        fileWrite.write('rfc=RandomForestClassifier()\n')
                        fileWrite.write(f'rfc_feature = RFE(rfc, n_features_to_select={n_feature[0]})\n')
                        fileWrite.write('rfc_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[rfc_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('rfc.fit(x_train,y_train)\n')
                        fileWrite.write('rfctd=rfc.predict(x_test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, rfctd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        fileWrite.write('rfc=RandomForestRegressor()\n')
                        fileWrite.write(f'rfc_feature = RFE(rfc, n_features_to_select={n_feature[1]})\n')
                        fileWrite.write('rfc_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[rfc_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('rfc.fit(x_train,y_train)\n')
                        fileWrite.write('rfctd=rfc.predict(x_test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, rfctd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        fileWrite.write('rfc=RandomForestRegressor()\n')
                        fileWrite.write(f'rfc_feature = RFE(rfc, n_features_to_select={n_feature[2]})\n')
                        fileWrite.write('rfc_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[rfc_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('rfc.fit(x_train,y_train)\n')
                        fileWrite.write('rfctd=rfc.predict(x_test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, rfctd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        fileWrite.write('rfc=RandomForestRegressor()\n')
                        fileWrite.write(f'rfc_feature = RFE(rfc, n_features_to_select={n_feature[3]})\n')
                        fileWrite.write('rfc_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[rfc_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('rfc.fit(x_train,y_train)\n')
                        fileWrite.write('rfctd=rfr.predict(x_test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, rfctd))\n\n\n')
                    else:
                        pass

                elif sc.index(max(sc))==3:
                    if select_fea_accr.index(max(select_fea_accr))==0:
                        fileWrite.write('xgb = XGBRegressor()\n')
                        fileWrite.write(f'xgb_feature = RFE(xgb, n_features_to_select={n_feature[0]})\n')
                        fileWrite.write('xgb_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[xgb_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('xgb.fit(x_train,y_train)\n')
                        fileWrite.write('xgbtd=xgb.predict(x_test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, xgbtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==1:
                        fileWrite.write('xgb = XGBRegressor()\n')
                        fileWrite.write(f'xgb_feature = RFE(xgb, n_features_to_select={n_feature[1]})\n')
                        fileWrite.write('xgb_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[xgb_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('xgb.fit(x_train,y_train)\n')
                        fileWrite.write('xgbtd=xgb.predict(x_test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, xgbtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==2:
                        fileWrite.write('xgb = XGBRegressor()\n')
                        fileWrite.write(f'xgb_feature = RFE(xgb, n_features_to_select={n_feature[2]})\n')
                        fileWrite.write('xgb_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[xgb_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('xgb.fit(x_train,y_train)\n')
                        fileWrite.write('xgbtd=xgb.predict(x_test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, xgbtd))\n\n\n')
                    elif select_fea_accr.index(max(select_fea_accr))==3:
                        fileWrite.write('xgb = XGBRegressor()\n')
                        fileWrite.write(f'xgb_feature = RFE(xgb, n_features_to_select={n_feature[3]})\n')
                        fileWrite.write('xgb_feature.fit(x_train, y_train)\n')
                        fileWrite.write('selected_features = x_train.columns[xgb_feature.support_]\n')
                        fileWrite.write('x_train = x_train[selected_features]\n')
                        fileWrite.write('x_test = x_test[selected_features]\n\n')


                        fileWrite.write('xgb.fit(x_train,y_train)\n')
                        fileWrite.write('xgbtd=xgb.predict(x_test)\n')
                        fileWrite.write('print("accuracy",accuracy_score(y_Test, xgbtd))\n\n\n')
                    else:
                        pass
                else:
                    pass
           
        else:
            pass
        
        
        
        ################################################################################
        ################################# Parameter Tuning #############################
        ################################################################################
        if "option8" in additional_options:
            if sc.index(max(sc))==0:
                
                svcparam=[{'C': [3, 5, 7, 9, 0.1, 1, 10, 100, 1000],
                        "kernel":['rbf','linear','sigmoid'],
                        'gamma': [0, 0.1, 0.2,0.03,0.3 ]}
                        ]
                fileWrite.write('svcparam=[{"C": [3, 5, 7, 9, 0.1, 1, 10, 100, 1000],"kernel":["rbf","linear","sigmoid"],"gamma":[0, 0.1, 0.2,0.03,0.3]}]\n\n')

                #Performing GridSearchCV
                fileWrite.write('#Performing GridSearchCV\n')
                svc=supportVectorClassifier() 
                fileWrite.write('svc=SVC()\n')
                svcGrid_search=GridSearchCV(estimator=svc,param_grid=svcparam, cv=9, n_jobs=-1)
                fileWrite.write('svcGrid_search=GridSearchCV(estimator=svc,param_grid=svcparam, cv=9, n_jobs=-1)\n')
                svcGrid_search.fit(x_Train,y_Train)
                fileWrite.write('svcGrid_search.fit(x_Train,y_Train)\n\n')

                #Getting the best score
                fileWrite.write('#Getting the best score\n')
                svcaccuracy=svcGrid_search.best_score_
                fileWrite.write('print("best score",svcGrid_search.best_score_)\n\n')
                
                

                fileWrite.write('#Getting the best parameters\n')
                
                fileWrite.write('print(f"best parameters:{svcGrid_search.best_params_}")\n\n\n')
                
                fileWrite.write("###########################################################\n")
                fileWrite.write("#********** Training Model using best parameters **********\n")
                fileWrite.write("###########################################################\n\n")
                svc = supportVectorClassifier(C=svcGrid_search.best_params_['C'],gamma=svcGrid_search.best_params_['gamma'],kernel=svcGrid_search.best_params_['kernel']) 
                
                fileWrite.write(f"svc = SVC(C=svcGrid_search.best_params_['C'],gamma=svcGrid_search.best_params_['gamma'],kernel=svcGrid_search.best_params_['kernel'])\n")
                svc.fit(x_Train,y_Train)
                fileWrite.write('svc.fit(x_Train,y_Train)\n')
                
                fileWrite.write('print("score:",svc.score(x_Test,y_Test))\n\n\n')
            ##############################################################
            #################### KNN Tuning ##############################
            ##############################################################    
            elif sc.index(max(sc))==1:
                
                knnparam=[
                    {'n_neighbors': [3, 5, 7, 9,13,15],
                    'weights': ['uniform', 'distance'],'metric':['manhattan','chebyshev','minkowski']}
                    ]
                fileWrite.write('knnparam=[{"n_neighbors": [3, 5, 7, 9,13,15],"weights": ["uniform", "distance"],"metric":["manhattan","chebyshev","minkowski"]}]\n\n')

                #Performing GridSearchCV
                fileWrite.write('#Performing GridSearchCV\n')
                knn=kNeighborsClassifier() 
                fileWrite.write('knn=KNeighborsClassifier()\n')
                knnGrid_search=GridSearchCV(estimator=knn,param_grid=knnparam, cv=9, n_jobs=-1)
                fileWrite.write('knnGrid_search=GridSearchCV(estimator=knn,param_grid=knnparam, cv=9, n_jobs=-1)\n')
                knnGrid_search.fit(x_Train,y_Train)
                fileWrite.write('knnGrid_search.fit(x_Train,y_Train)\n\n')

                #Getting the best score
                fileWrite.write('#Getting the best score\n')
                knnaccuracy=knnGrid_search.best_score_
                fileWrite.write('print("best score",knnGrid_search.best_score_)\n\n')
                
                
                fileWrite.write('#Getting the best parameters\n')
                
                fileWrite.write('print(f"best parameters:{knnGrid_search.best_params_}")\n\n\n')
                
                fileWrite.write("###########################################################\n")
                fileWrite.write("#********** Training Model using best parameters **********\n")
                fileWrite.write("###########################################################\n\n")
                knn = kNeighborsClassifier(leaf_size=knnGrid_search.best_params_['leaf_size'],n_neighbors=knnGrid_search.best_params_['n_neighbors'],weights=knnGrid_search.best_params_['weights']) 
                
                fileWrite.write(f"knn = KNeighborsClassifier(leaf_size=knnGrid_search.best_params_['leaf_size'],n_neighbors=knnGrid_search.best_params_['n_neighbors'],weights=knnGrid_search.best_params_['weights'])\n")
                knn.fit(x_Train,y_Train)
                fileWrite.write('knn.fit(x_Train,y_Train)\n')
                
                fileWrite.write('print("score:",knn.score(x_Test,y_Test))\n\n\n')
            ##############################################################
            #################### RFR Tuning ##############################
            ##############################################################    
            elif sc.index(max(sc))==2:  
                
                rfcparam=[{"n_estimators":[100,150,170,290],
                    "criterion":["gini", "entropy"],
                    "max_depth":[5,10, None],
                    "min_samples_split":[2, 6, 10],"min_samples_leaf":[1, 3, 4],
                    }
                    ]
                fileWrite.write('rfcparam=[{"n_estimators":[100,150,170,290],"criterion":["gini", "entropy"],"max_depth":[5,10, None],"min_samples_split":[2, 6, 10],"min_samples_leaf":[1, 3, 4]}]\n\n')

                #Performing GridSearchCV
                fileWrite.write('#Performing GridSearchCV\n')
                rfc=randomForestClassifier() 
                fileWrite.write('rfc=RandomForestClassifier()\n')
                rfcGrid_search=GridSearchCV(estimator=rfc,param_grid=rfcparam, cv=9, n_jobs=-1)
                fileWrite.write('rfcGrid_search=GridSearchCV(estimator=rfc,param_grid=rfcparam, cv=9, n_jobs=-1)\n')
                rfcGrid_search.fit(x_train,y_train)
                fileWrite.write('rfcGrid_search.fit(x_train,y_train)\n\n')

                #Getting the best score
                fileWrite.write('#Getting the best score\n')
                rfcaccuracy=rfcGrid_search.best_score_
                fileWrite.write('print("best score",rfcGrid_search.best_score_)\n\n')
                
                

                #Getting the best parameters
                fileWrite.write('#Getting the best parameters\n')
                
                fileWrite.write('print(f"best parameters:{rfcGrid_search.best_params_}")\n\n\n')
                
                fileWrite.write("###########################################################\n")
                fileWrite.write("#********** Training Model using best parameters **********\n")
                fileWrite.write("###########################################################\n\n")
                rfc = randomForestClassifier(n_estimators=rfcGrid_search.best_params_['n_estimators'],criterion=rfcGrid_search.best_params_['criterion'],max_depth=rfcGrid_search.best_params_['max_depth'],
                    min_samples_split=rfcGrid_search.best_params_['min_samples_split'],min_samples_leaf=rfcGrid_search.best_params_['min_samples_leaf'])
                
                fileWrite.write(f"rfc = RandomForestClassifier(n_estimators=rfcGrid_search.best_params_['n_estimators'],criterion=rfcGrid_search.best_params_['criterion'],max_depth=rfcGrid_search.best_params_['max_depth'],min_samples_split=rfcGrid_search.best_params_['min_samples_split'],min_samples_leaf=rfcGrid_search.best_params_['min_samples_leaf']\n")
                rfc.fit(x_train,y_train)
                fileWrite.write('rfc.fit(x_train,y_train)\n')
                
                fileWrite.write('print("score:",rfc.score(x_test,y_test))\n\n\n')
            ##############################################################
            #################### XGBoost Tuning ##########################
            ##############################################################    
            elif sc.index(max(sc))==3:
                
                xgbparam=[
                    
                    { 'min_child_weight': [2, 5, 8],
                    'gamma': [0, 0.1, 0.2 ],
                    'subsample': [0.6, 0.8],
                    'colsample_bytree': [0.6, 0.8],
                        'max_depth': [3,  5, 7, 9],
                        "learning_rate":[ 0.1, 0.01,  0.2,  0.3],
                        "n_estimators":[100,150, 250, 400 ]}
                        ]
                fileWrite.write('xgbparam=[{ "min_child_weight": [2, 5, 8],"gamma": [0, 0.1, 0.2 ],"subsample": [0.6, 0.8],"colsample_bytree": [0.6, 0.8],"max_depth": [3, 5,7,9],"learning_rate":[0.1, 0.01,  0.2,  0.3],"n_estimators":[100,150, 250, 400]}]\n\n')

                #Performing GridSearchCV
                fileWrite.write('#Performing GridSearchCV\n')
                xgb = xGBClassifier()
                fileWrite.write('xgb = XGBClassifier()\n')
                xgbGrid_search=GridSearchCV(estimator=xgb,param_grid=xgbparam, cv=9, n_jobs=-1)
                fileWrite.write('xgbGrid_search=GridSearchCV(estimator=xgb,param_grid=xgbparam, cv=9, n_jobs=-1)\n')
                xgbGrid_search.fit(x_train,y_train)
                fileWrite.write('xgbGrid_search.fit(x_train,y_train)\n\n')

                #Getting the best score
                fileWrite.write('#Getting the best score\n')
                xgbaccuracy=xgbGrid_search.best_score_
                fileWrite.write('print("best score",xgbGrid_search.best_score_)\n\n\n')
                
                
                #Getting the best parameters
                fileWrite.write('#Getting the best parameters\n')
                
                fileWrite.write('print(f"best parameters:{xgbGrid_search.best_params_}")\n\n\n')
                
                fileWrite.write("###########################################################\n")
                fileWrite.write("#********** Training Model using best parameters **********\n")
                fileWrite.write("###########################################################\n\n")
                xgb = xGBClassifier(colsample_bytree=rfrGrid_search.best_params_['colsample_bytree'],gamma=rfrGrid_search.best_params_['gamma'],learning_rate=rfrGrid_search.best_params_['learning_rate'],
                    max_depth=rfrGrid_search.best_params_['max_depth'],min_child_weight=rfrGrid_search.best_params_['min_child_weight'],n_estimators=rfrGrid_search.best_params_['n_estimators'],
                    subsample=rfrGrid_search.best_params_['subsample'])

                fileWrite.write(f"xgb = XGBClassifier(colsample_bytree=rfrGrid_search.best_params_['colsample_bytree'],gamma=rfrGrid_search.best_params_['gamma'],learning_rate=rfrGrid_search.best_params_['learning_rate'],max_depth=rfrGrid_search.best_params_['max_depth'],min_child_weight=rfrGrid_search.best_params_['min_child_weight'],n_estimators=rfrGrid_search.best_params_['n_estimators'],subsample=rfrGrid_search.best_params_['subsample'])\n")
                xgb.fit(x_train,y_train)
                fileWrite.write('xgb.fit(x_train,y_train)\n')
                
                fileWrite.write('print("score:",xgb.score(x_test,y_test))\n\n\n')

            else:
                pass
            
        else:
            pass

        
        
        if "option9" in additional_options:
            fileWrite.write("######################################\n")
            fileWrite.write("#****** Model evaluation ************\n")
            fileWrite.write("######################################\n\n")
            if sc.index(max(sc))==0:
                # make predictions on the testing set
                fileWrite.write('# make predictions on the testing set\n')
                fileWrite.write('y_pred = svc.predict(x_Test)\n\n')
                
                # calculate the evaluation metrics
                fileWrite.write('# calculate the evaluation metrics\n\n')
                fileWrite.write('#accuracy\n')
                fileWrite.write('print("accuracy",accuracy_score(y_Test, y_pred))\n\n')
                
                
                fileWrite.write('#precision\n')
                fileWrite.write('print("precision",precision_score(y_Test, y_pred))\n\n')
                
                
                fileWrite.write('#Recall\n')
                fileWrite.write('print("recall",recall_score(y_Test, y_pred))\n\n')
               
                
                fileWrite.write('f1 score\n')
                fileWrite.write('print("f1_score",f1_score(y_Test, y_pred))\n\n')
               
                
                fileWrite.write('confusion matrix\n')
                fileWrite.write('print("confusion matrix",confusion_matrix(y_Test, y_pred))\n\n\n')
                
                
                
            elif sc.index(max(sc))==1:
                # make predictions on the testing set
                fileWrite.write('# make predictions on the testing set\n')
                fileWrite.write('y_pred = knn.predict(x_Test)\n\n')

                # calculate the evaluation metrics
                fileWrite.write('# calculate the evaluation metrics\n\n')
                fileWrite.write('#accuracy\n')
                fileWrite.write('print("accuracy",accuracy_score(y_Test, y_pred))\n\n')
                
                
                fileWrite.write('#precision\n')
                fileWrite.write('print("precision",precision_score(y_Test, y_pred))\n\n')
                
                
                fileWrite.write('#Recall\n')
                fileWrite.write('print("recall",recall_score(y_Test, y_pred))\n\n')
               
                
                fileWrite.write('f1 score\n')
                fileWrite.write('print("f1_score",f1_score(y_Test, y_pred))\n\n')
               
                
                fileWrite.write('confusion matrix\n')
                fileWrite.write('print("confusion matrix",confusion_matrix(y_Test, y_pred))\n\n\n')
                
            elif sc.index(max(sc))==2:
                # make predictions on the testing set
                fileWrite.write('# make predictions on the testing set\n')
                fileWrite.write('y_pred = rfc.predict(x_test)\n\n')

                # calculate the evaluation metrics
                fileWrite.write('# calculate the evaluation metrics\n\n')
                fileWrite.write('#accuracy\n')
                fileWrite.write('print("accuracy",accuracy_score(y_test, y_pred))\n\n')
                
                
                fileWrite.write('#precision\n')
                fileWrite.write('print("precision",precision_score(y_test, y_pred))\n\n')
                
                
                fileWrite.write('#Recall\n')
                fileWrite.write('print("recall",recall_score(y_test, y_pred))\n\n')
               
                
                fileWrite.write('f1 score\n')
                fileWrite.write('print("f1_score",f1_score(y_test, y_pred))\n\n')
               
                
                fileWrite.write('confusion matrix\n')
                fileWrite.write('print("confusion matrix",confusion_matrix(y_test, y_pred))\n\n\n')
            elif sc.index(max(sc))==3:
                # make predictions on the testing set
                fileWrite.write('# make predictions on the testing set\n')
                fileWrite.write('y_pred = xgb.predict(x_test)\n\n')

                # calculate the evaluation metrics
                fileWrite.write('# calculate the evaluation metrics\n\n')
                fileWrite.write('#accuracy\n')
                fileWrite.write('print("accuracy",accuracy_score(y_test, y_pred))\n\n')
                
                
                fileWrite.write('#precision\n')
                fileWrite.write('print("precision",precision_score(y_test, y_pred))\n\n')
                
                
                fileWrite.write('#Recall\n')
                fileWrite.write('print("recall",recall_score(y_test, y_pred))\n\n')
               
                
                fileWrite.write('f1 score\n')
                fileWrite.write('print("f1_score",f1_score(y_test, y_pred))\n\n')
               
                
                fileWrite.write('confusion matrix\n')
                fileWrite.write('print("confusion matrix",confusion_matrix(y_test, y_pred))\n\n\n')
            
        else:
            pass 
        
            
    elif request.form['operation'] == "text_classification":
        pass
    else:
        pass
    
    fileWrite.close()
    
   
    return render_template('result.html')
    
@app.route('/download')
def download_file():
    filename = 'ml-code.py'
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=False)
