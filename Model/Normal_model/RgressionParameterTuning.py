from . import RegressionConfig as Configure
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from hyperopt import fmin,tpe,STATUS_OK,Trials
from sklearn.preprocessing import scale,normalize
from sklearn.model_selection import cross_val_score



class ModelChoice:
    def __init__(self,feature,label):
        self.feature=feature
        self.label=label
        self.bestTemp=None


    def regressor_hyperopt_train_test(self,params):
        if 'scale' in params:
            if params['scale']==1:
                self.feature=scale(self.feature)
            del params['scale']
        if 'normalize' in params:
            if params['normalize']==1:
                self.feature=normalize(self.feature)
            del params['normalize']
        t=params['type']
        del params['type']
        if  t=='RandomForestRegressor':
            regressor=RandomForestRegressor(**params)
        elif t=='GradientBoostingRegressor':
            regressor=GradientBoostingRegressor(**params)
        elif t == 'SVR':
            kernel_cfg=params.pop('kernel')
            kernel_name=kernel_cfg.pop('name')
            params.update(kernel_cfg)
            params['kernel']=kernel_name
            regressor=SVR(**params)
        else:
            return 0
        return cross_val_score(regressor,self.feature,self.label,scoring='neg_mean_squared_error')

    def lossfunction(self,params):
        paramsTemp=params.copy()
        tempResult=self.regressor_hyperopt_train_test(params)
        mse=-tempResult.mean()
        std=tempResult.std()
        if self.bestTemp==None:
            self.bestTemp=mse
        if mse<self.bestTemp:
            self.bestTemp = mse
            print("the new best mse is {}  std is {}  using model {} parameters {}".format(mse,std, paramsTemp['type'], paramsTemp))
        return {'loss':mse,'status':STATUS_OK}


    def regressorChoice(self):
        trials=Trials()
        best=fmin(fn=self.lossfunction,algo=tpe.suggest,space=Configure.RegressorSpace,max_evals=Configure.Parameter_tunning_max_evals,trials=trials)
        print("the best mse is {}".format(best))