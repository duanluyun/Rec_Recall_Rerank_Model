from . import ClassifierConfig as Configure
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import  BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,KFold
from hyperopt import fmin,tpe,STATUS_OK,Trials
from sklearn.preprocessing import scale,normalize


class Modelchoice:
    def __init__(self,features,label):
        self.features=features
        self.label=label
        self.bestTemp=None


    def classfier_hyperopt_train_test(self,params):
        if 'scale' in params:
            if params['scale']==1:
                self.features=scale(self.features)
            del params['scale']
        if 'normalize' in params:
            if params['normalize']==1:
                self.features=normalize(self.features)
            del params['normalize']
        t=params['type']
        del params['type']
        if  t=='RandomForestClassifier':
            clf=RandomForestClassifier(**params)
        elif t=='KNeighborsClassifier':
            clf=KNeighborsClassifier(**params)
        elif t=='BernoulliNB':
            clf=BernoulliNB(**params)
        elif t=='SVC':
            kernel_cfg = params.pop('kernel')
            kernel_name = kernel_cfg.pop('name')
            params.update(kernel_cfg)
            params['kernel'] = kernel_name
            clf = SVC(**params)
        else:
            return 0
        return cross_val_score(clf,self.features,self.label).mean()

    def lossfunction(self,params):
        paramsTemp=params.copy()
        acc=self.classfier_hyperopt_train_test(params)
        if self.bestTemp==None:
            self.bestTemp=acc
        if acc>self.bestTemp:
            self.bestTemp=acc
            print("the new best acc is {}  using model {} parameters {}".format(acc,paramsTemp['type'],paramsTemp))
        return {'loss':-acc,'status':STATUS_OK}

    def classfierChoice(self):
        trials=Trials()
        best=fmin(fn=self.lossfunction,space=Configure.classfiertSpace,algo=tpe.suggest,max_evals=Configure.Parameter_tunning_max_evals,trials=trials)
        print(best)