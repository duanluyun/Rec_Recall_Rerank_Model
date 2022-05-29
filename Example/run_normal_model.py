import warnings
from Model.Normal_model import ClassificationParameterTuning
from Model.Normal_model import RgressionParameterTuning
from sklearn import datasets



def run_master(model,features,labels):
    if    model=='Classification':
        classfier=ClassificationParameterTuning.Modelchoice(features,labels)
        classfier.classfierChoice()
    elif  model=='Regression':
        regressor=RgressionParameterTuning.ModelChoice(features,labels)
        regressor.regressorChoice()
    else:
        print("please choice the correct model")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    dataTemp = datasets.load_iris()
    features = dataTemp.data
    label = dataTemp.target

    run_master('Classification',features,label)
