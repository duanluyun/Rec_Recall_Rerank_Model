from  hyperopt import hp



#Hyperopt的最大迭代次数
Parameter_tunning_max_evals=1500

RegressorSpace=hp.choice(
    'Regressor',[
        {
            'type':'RandomForestRegressor',
            'max_depth':hp.choice('rf_max_depth',range(1,50)),
            'n_estimators':hp.choice('rf_n_estimators',range(1,500)),
            'criterion':hp.choice('rf_criterion',['mse','mae']),
            'scale':hp.choice('rf_scale',[0,1]),
            'normalize':hp.choice('rf_normalize',[0,1])
        },
        {
            'type':'GradientBoostingRegressor',
             'max_depth':hp.choice('gbdt_max_depth',range(1,50)),
             'n_estimators':hp.choice('gbdt_n_estimators',range(1,500)),
            'learning_rate':hp.uniform('gbdt_learning_rate',0,1),
            'min_samples_split':hp.choice('gbdt_min_samples_split',range(1,500,2))
        },
        {
             'type':'SVR',
             'scale':hp.choice('SVR_scale',[0,1]),
             'normalize':hp.choice('SVR_normalize',[0,1]),
             'C':hp.uniform('SVR_C',0,1),
             'shrinking':hp.choice('SVR_shrinking',[True,False]),
             'kernel':hp.choice('SVR_kernel',[
                 {
                  'name':'rbf',
                  'gamma':hp.choice('SVR_rbf_gamma',['auto',hp.uniform('SVR_rbf_gamma_uniform',0.0001,8)])
                 },
                 {
                     'name':'linear'
                 },
                 {
                     'name':'sigmod',
                     'gamma':hp.choice('SVR_sigmoid_gamma',['auto',hp.uniform('SVR_sigmoid_gamma_uniform',0.0001,8)]),
                     'coef0':hp.uniform("SVR_sigmoid_coef0",0,10)
                 },
                 {
                  'name':'ploy',
                  'gamma': hp.choice('SVR_ploy_gamma', ['auto', hp.uniform('SVR_ploy_gamma_uniform', 0.0001, 8)]),
                  'coef0':hp.uniform("SVR_ploy_coef0",0,10),
                  'degree':hp.uniformint("SVR_ploy_degree",1,5),
                 }
             ])
        }
    ]
)