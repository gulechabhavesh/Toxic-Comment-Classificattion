class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
result = pd.DataFrame.from_dict({'id': test['Id']})
models = []

with open("file_name.pckl", "rb") as f:
    while True:
        try:
            models.append(pickle.load(f))
        except EOFError:
            break
i=0
for class_name in class_names:

    #uncomment below line if models are logistic,SGD,random forest
    #result[class_name] = models[i].predict_proba(test_features)[:, 1]

    #uncomment below line if models are ridge,xgboost
    #result[class_name] = models[i].predict(test_features)[:, 1]
    i=i+1
result.to_csv('file_name.csv', index=False)
