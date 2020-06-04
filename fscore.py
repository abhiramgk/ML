import pandas as pd

def fscore(test_data,predictions):
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(len(test_data)):
        if test_data[i] == 2 and predictions[i]==2:
            tp += 1
        elif test_data[i] == 4 and predictions[i]==4:
            tn +=1
        elif test_data[i] == 2 and predictions[i]==4:
            fn +=1
        else:
            fp +=1  
    accuracy= ((tp+tn)/(tp+tn+fp+fn))*100
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    return 2*((precision*recall)/(precision+recall))*100