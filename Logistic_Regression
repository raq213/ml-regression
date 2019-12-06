import pandas 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#dataset of the candidates
#in admission 0 is the false value (not admitted) and 1 true value (admitted)
candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'workExperience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admission': [1,1,1,1,1,1,0,1,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }

data = pandas.DataFrame(candidates,columns= ['gmat', 'gpa','workExperience','admission'])

X = data[['gmat', 'gpa','workExperience']]
y = data['admission']  

#75% is the train mode and rest 25% are test results
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)  

logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)

#y_pred=logistic_regression.predict(X_test)


#new cantidates to check 

#data set of new candidates to predict admission
new_candidates = {'gmat': [590,740,680,610,710],
                  'gpa': [2,3.7,3.3,2.3,3],
                  'workExperience': [3,4,6,1,5]
                  }

newData = pandas.DataFrame(new_candidates,columns= ['gmat', 'gpa','workExperience'])
y_pred=logistic_regression.predict(newData)

#test dataset (without the actual outcome)
#print (X_test)

#predicted values
#print (y_pred) 

#showing tha data of new candidates 
print (newData)

#showing the predictions for new candidates 
print (y_pred)
