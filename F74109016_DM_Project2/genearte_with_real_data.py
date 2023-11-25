from faker import Faker
from sklearn.datasets import load_iris
from sklearn import tree,svm
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score,  classification_report,RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import dtreeviz
import csv
import pandas as pd
from subprocess import call
from IPython.display import Image
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
def drawDataset(Dataset):
  # draw every attribute's bar graph
  for i in range(0,16):
    plt.figure(figsize=(10,6))
    plt.title(Dataset.columns[i],fontsize=50)
    plt.xlabel(Dataset.columns[i])
    plt.ylabel('count')
    plt.hist(Dataset[Dataset.columns[i]],bins=40,color='orange',edgecolor='black',alpha=0.7)
    plt.savefig("Data/"+str(i)+"_"+Dataset.columns[i]+".png")
    plt.show()

features = [
  "gender", "height", "weight", "education",
   "age", "working_time_hour",
  "working_satisfaction", "salary", 
  "personal_life", "frequency_of_attend_club",
  "Bonus_month", "working_participation",
  "relationship","change_job","position","health"]
def will_leave(attribute):
    probOfLeave = 0
    
    # working_satisfaction <= 2.5 -> will leave
    if(attribute['working_satisfaction'] <= 2.5):
      probOfLeave += 0.2
    # position <= 4.5 -> will leave
    if(attribute['position'] <= 4.5): 
      probOfLeave += 0.2
    # personal_life <= 1.5 -> will leave 
    if(attribute['personal_life'] <= 1.5):
      probOfLeave += 0.2
    #Change_job <= 0.5 -> will leave
    if(attribute['change_job'] <= 0.5):
      probOfLeave += 0.2
    # Working_participation <= 2.5 -> will leave
    if(attribute['working_participation'] <= 2.5):
      probOfLeave += 0.2

    # working_satisfaction > 2.5 -> not leave
    if(attribute['working_satisfaction'] > 2.5):
      probOfLeave -= 0.2
    # Personal_life > 1.5 -> not leave
    if(attribute['personal_life'] > 1.5):
      probOfLeave -= 0.2
    #Salary > 62600 -> not leave 
    if(attribute['salary'] > 62600):
      probOfLeave -= 0.2
    #Working_time_hour > 13.5 -> not leave
    if(attribute['working_time(hour)'] > 13.5):
      probOfLeave -= 0.2
    #Frequency_of_attend_club >= 3.4 -> not leave
    if(attribute['frequency_of_attend_club'] >= 3.4):
      probOfLeave -= 0.2


    if probOfLeave >= 1 : probOfLeave = 0.9
    if probOfLeave <= 0 : probOfLeave = 0.1
    if np.random.random() < probOfLeave:
        return True
    else:
        return False
    
def generateDataSet(number):
  Data = Faker()
  Dataset ,target= [],[]
  features = ["gender","height","weight","education","age",
              "working_time_hour","working_satisfaction","salary", 
              "personal_life","frequency_of_attend_club","Bonus_month","working_participation",
              "relationship","change_job","position","health","target"]
  leave_num = 0
  not_leave_num = 0
  for _ in range(int(number)):
    attribute = {}
    attribute_list = []
    # 一、性別（0 : 男生, 1 : 女生） 
    attribute['gender'] = Data.random_int(0,1)
    # 二、身高（男生平均：175 cm , 女生平均：160 cm，標準差20以高斯分佈生成） 
    # 三、體重（男生平均：65 kg , 女生平均：55 kg，標準差10以高斯分佈生成） 
    if(attribute['gender'] == 0):
      attribute['height'] = np.random.normal(175, 20)
      attribute['weight'] = np.random.normal(65, 20)
    else:
      attribute['height'] = np.random.normal(160, 20)
      attribute['weight'] = np.random.normal(55, 20)
    # 四、教育（1：小學以下, 2：國中, 3：高中職, 4：大專院校, 5：研究所以上） 
    attribute['education'] = Data.random_int(1,5)
    # 五、年齡（設定平均年齡：45歲，標準差20，以高斯分佈生成）
    attribute['age'] = int(np.random.normal(45, 20))
    # 六、每天平均上班時間（0∼24 小時，每天平均:9hr，標準差3以高斯分佈生成）
    attribute['working_time(hour)'] = int(np.random.normal(9, 3))
    # 七、工作滿意度（1 : 每天無精打采 ,2 : 開會打哈欠很多次, 3 : 回email速度很快, 
    # 4 : 每周固定回報工作進度, 5 : 與同事合作討論工作,  1∼5均勻隨機生成）
    attribute['working_satisfaction'] = Data.random_int(1,5)
    # 八、薪資水平（平均 : 40000,標準差15000以高斯分佈生成） 
    attribute['salary'] = int(np.random.normal(40000, 15000))
    # 九、個人生活狀況（1：有小孩，2：沒有小孩，3: 有結婚，4: 沒有結婚） 
    attribute['personal_life'] = Data.random_int(1,4)
    # 十、參加公司社團頻率（1：幾乎不去，2：偶爾去，3：經常去，4：每天報到） 
    attribute['frequency_of_attend_club'] = Data.random_int(1,4)
    # 十一、距離上次加薪幾個月? ( 0 ~ 24 ，以均勻分配隨機生成)
    attribute['Bonus_month'] = Data.random_int(0,24)
    # 十二、員工參與度(1: 會跟同事一起加班，2: 會跟同事一起吃飯，
    # 3: 會參加公司活動，4: 出國會帶點心，5: 會接下外務)
    attribute['working_participation'] = Data.random_int(1,5)
    # 十三、上級主管關係（1 ∼ 10，數字越高，代表關係越好）
    attribute['relationship'] = Data.random_int(1,10)
    # 十四、出社會後跳槽平均頻率（0 : 一年內, 1 : 兩年內 2: 五年內, 3: 沒換過公司） 
    attribute['change_job'] = Data.random_int(0,3)
    # 十五. 公司職務 （1:助理,2:職員,3:組長,4:經理,5:處長,6:總裁,7董事長）
    # 機率為 (0.1,0.2,0.3,0.2,0.1,0.05,0.05)
    attribute['position'] = np.random.choice([1,2,3,4,5,6,7], p=[0.1,0.2,0.3,0.2,0.1,0.05,0.05])
    # 十六. 健康度 (1:健康,2:生病,3:住院,4:生病住院)
    # 機率為 (0.7,0.1,0.1,0.1)
    attribute['health'] = np.random.choice([1,2,3,4], p=[0.7,0.1,0.1,0.1])
    for key, value in attribute.items():
      print(key)
      attribute_list.append(value)
    
    Dataset.append(attribute_list)
    leave = will_leave(attribute)
    if(leave == True): 
      print("離職")
      leave_num +=1
    else:              
      not_leave_num +=1
    # draw leave and not leave bar graph
    target.append(leave)
  print("leave : ",leave_num)
  print("not leave : ",not_leave_num)
  

  
  for  i in range(len(Dataset)):
    Dataset[i].append(target[i])
  with open('output_with_realdata.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(features)
    writer.writerows(Dataset)

def readDataset(filename):
  Dataset = []
  with open('output_with_realdata.csv', newline='') as csvfile:
    data = csv.reader(csvfile, delimiter='\t')
    csv_read_list = list(data)
    first = True
    for row in csv_read_list:
      if first ==  False:       
        row[0] = int(row[0])
        row[1] = float(row[1])
        row[2] = float(row[2])
        row[3] = int(row[3])
        row[4] = int(row[4])
        row[5] = int(row[5])
        row[6] = int(row[6])
        row[7] = int(row[7])
        row[8] = int(row[8])
        row[9] = int(row[9])
        row[10] = int(row[10])
        row[11] = int(row[11])
        row[12] = int(row[12])
        row[13] = int(row[13])
        row[14] = int(row[14])
        row[15] = int(row[15])
    
        if(row[16] == 'True'): row[16] = True
        else:                  row[16] = False
        Dataset.append(row)
      else :
        first = False    
  datanew = DataFrame(Dataset)

  datanew.rename(columns={  0 : 'gender',
                            1 : 'height',
                            2 : 'weight',
                            3 : 'education',
                            4 : 'age',
                            5 : 'working_time(hour)',
                            6 : 'working_satisfaction',
                            7 : 'salary',
                            8 : 'personal_life',
                            9 : 'frequency_of_attend_club',
                            10 : 'Bonus_month',
                            11 : 'working_participation',
                            12 : 'relationship',
                            13 : 'change_job',
                            14 : 'position',
                            15 : 'health',
                            16 : 'target'},inplace = True)
  return datanew

def evaluate_model(dt_classifier,X_train, X_test, Y_train, Y_test):
    print("Train Accuracy :", accuracy_score(Y_train, dt_classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    predict_train = dt_classifier.predict(X_train)
    print(confusion_matrix(Y_train, predict_train))
    print(classification_report(Y_train,predict_train))
    print("-"*50)
    predict_test = dt_classifier.predict(X_test)
    print("Test Accuracy :", accuracy_score(Y_test, predict_test))
    print("Test Confusion Matrix:")
    print(confusion_matrix(Y_test, predict_test))
    print(classification_report(Y_test,predict_test))

################# DecisionTree #############################
def DecisionTree( X_train, X_test, Y_train, Y_test):
  model = tree.DecisionTreeClassifier(max_leaf_nodes = 50)
  model.fit(X_train, Y_train)
  evaluate_model(model,X_train, X_test, Y_train, Y_test)
  
  export_graphviz(model, out_file='tree_decision_classifier.dot', 
                feature_names = features, class_names = ['leave', 'not leave'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)
  call(['dot', '-Tpng', '/home/weihsin/projects/DM/HW2/tree_decision_classifier.dot', '-o', 'tree_decision_classifier.png', '-Gdpi=6000'])
  Image(filename = 'tree_decision_classifier.png')
  return model
################# KNN #############################
def KNN( X_train, X_test, Y_train, Y_test):
  #從k=1開始測試利用for loop來建立迴圈，選擇k值
  error_rate = []
  for i in range(1,60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != Y_test))
  smallest = 1000
  index = 0
  for i in range(0,59):
    if(smallest > error_rate[i]):
      smallest = error_rate[i]
      index = i+1
  plt.figure(figsize=(10,6))
  plt.plot(range(1,60),error_rate,color='orange', linestyle='solid', marker='o', markersize=10)
  plt.title('Error Rate vs. K Value')
  plt.xlabel('K')
  plt.ylabel('Error Rate')
  plt.savefig('KNN_error_rate.png')
  print("smallest error rate : ",smallest)
  print("index : ",index)
  knn = KNeighborsClassifier(n_neighbors=index)
  knn.fit(X_train,Y_train)
  evaluate_model(knn,X_train, X_test, Y_train, Y_test)
  return knn
################# LogisticRegression #############################
def LogisticRegressionrun( X_train, X_test, Y_train, Y_test):
  logmodel = LogisticRegression()
  logmodel.fit(X_train,Y_train)
  predictions = logmodel.predict(X_test)
  evaluate_model(logmodel,X_train, X_test, Y_train, Y_test)
  return logmodel
################# Naive Bayes #############################
def NaiveBayes( X_train, X_test, Y_train, Y_test):
  gnb = GaussianNB()
  gnb.fit(X_train,Y_train)
  predictions = gnb.predict(X_test)
  evaluate_model(gnb,X_train, X_test, Y_train, Y_test)
  return gnb
################# Random Forest #############################
def RandomForest( X_train, X_test, Y_train, Y_test):
  # 建立模型
  forest = RandomForestClassifier(n_estimators=100,max_leaf_nodes = 10)
  forest_fit = forest.fit(X_train, Y_train)
  fn=features
  cn=['leave', 'not leave']
  fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=2000)
  for index in range(0, 5):
      tree.plot_tree(forest_fit[index],
                    feature_names = fn, 
                    class_names=cn,
                    filled = True,
                    ax = axes[index])
      axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
  fig.savefig('Random_Forest_5trees.png')

  forest_predictions = forest_fit.predict(X_test)
  evaluate_model(forest_fit,X_train, X_test, Y_train, Y_test)
  return forest_fit
################# SVM #############################
def SVM( X_train, X_test, Y_train, Y_test):
  SVMclf = svm.SVC()
  SVMclf.fit(X_train, Y_train)
  evaluate_model(SVMclf,X_train, X_test, Y_train, Y_test)
  return SVMclf
################# MLP #############################
def MLP( X_train, X_test, Y_train, Y_test):
  from sklearn.neural_network import MLPClassifier
  mlp = MLPClassifier(hidden_layer_sizes=(6,12,24,12,6),
                        max_iter = 100,activation = 'logistic',batch_size=100,
                        solver = 'adam',learning_rate_init = 0.001)
  mlp.fit(X_train,Y_train)
  evaluate_model(mlp,X_train, X_test, Y_train, Y_test)
  return mlp
################# ROC curve #############################
def ROCcurve(X_train, X_test, Y_train, Y_test,DT_model,KNN_model,Log_model,NB_model,RF_model,SVM_model,MLP_model):
  #create ROC curve
  plt.figure(figsize=(10,6))
  DT_disp = RocCurveDisplay.from_estimator(DT_model, X_test, Y_test)
  KNN_disp = RocCurveDisplay.from_estimator(KNN_model, X_test, Y_test)
  Log_disp = RocCurveDisplay.from_estimator(Log_model, X_test, Y_test)
  NB_disp = RocCurveDisplay.from_estimator(NB_model, X_test, Y_test)
  RF_disp = RocCurveDisplay.from_estimator(RF_model, X_test, Y_test)
  SVM_disp = RocCurveDisplay.from_estimator(SVM_model, X_test, Y_test)
  MLP_disp = RocCurveDisplay.from_estimator(MLP_model, X_test, Y_test)
  ax = plt.gca()
  DT_disp.plot(ax=ax, alpha=0.8, color= 'cyan')
  ax = plt.gca()
  KNN_disp.plot(ax=ax, alpha=0.8, color= 'darkorange')
  ax = plt.gca()
  Log_disp.plot(ax=ax, alpha=0.8, color= 'lime')
  ax = plt.gca()
  NB_disp.plot(ax=ax, alpha=0.8, color= 'darkviolet')
  ax = plt.gca()
  SVM_disp.plot(ax=ax, alpha=0.8, color= 'deeppink')
  ax = plt.gca()
  RF_disp.plot(ax=ax, alpha=0.8, color= 'dodgerblue')

  
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.savefig("ROC.png")


if __name__ == '__main__':
  filename = "output.csv"
  generateDataSet(10000)
  Dataset = readDataset(filename)
  
  drawDataset(Dataset)
  scaler = StandardScaler()

  X = Dataset.drop('target',axis=1)
 
  y = Dataset['target']
  X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.3,random_state=0)
  print("###DecisionTree####################################################")
  DT_model = DecisionTree(  X_train, X_test, Y_train, Y_test)
  print("###KNN#############################################################")
  KNN_model = KNN( X_train, X_test, Y_train, Y_test)
  print("###LogisticRegression##############################################")
  Log_model = LogisticRegressionrun( X_train, X_test, Y_train, Y_test)
  print("###NaiveBayes#####################################################")
  NB_model = NaiveBayes( X_train, X_test, Y_train, Y_test)
  print("###RandomForest###################################################")
  RF_model = RandomForest( X_train, X_test, Y_train, Y_test)
  print("###SVM#############################################################")
  SVM_model = SVM( X_train, X_test, Y_train, Y_test)
  print("###MLP#############################################################")
  MLP_model = MLP( X_train, X_test, Y_train, Y_test)
  print("###ROC#############################################################")
  ROCcurve(X_train, X_test, Y_train, Y_test,DT_model,KNN_model,Log_model,NB_model,RF_model,SVM_model,MLP_model)

  