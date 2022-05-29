from  hyperopt import hp





Parameter_tunning_max_evals=1500

classfiertSpace=hp.choice(
    'classifier',[
        {
            'type':'RandomForestClassifier',
            'max_depth':hp.choice('max_depth',range(1,30)),
            'n_estimators':hp.choice('n_estimators',range(1,500)),
            'max_features':hp.choice('max_features',range(1,5)),
            'criterion':hp.choice('criterion',['gini','entropy']),
            'scale':hp.choice('rf_scale',[0,1]),
            'normalize':hp.choice('rf_normalize',[0,1])
        },
        {
            'type':'KNeighborsClassifier',
            'scale':hp.choice('knn_scale',[0,1]),
            'normalize':hp.choice('knn_normalize',[0,1]),
            'n_neighbors':hp.choice('n_neighbors',range(1,100))
        },
        {
            'type': 'SVC',
            'C': hp.uniform('SVC_C', 0, 1),
            'shrinking': hp.choice('SVC_shrinking', [True, False]),
            'kernel': hp.choice('SVC_kernel', [
                {
                    'name': 'rbf',
                    'gamma': hp.choice('svc_rbf_gamma', ['auto', hp.uniform('svc_rbf_gamma_uniform', 0.0001, 8)])
                },
                {
                    'name': 'linear'
                },
                {
                    'name': 'sigmod',
                    'gamma': hp.choice('svc_sigmoid_gamma',
                                       ['auto', hp.uniform('svc_sigmoid_gamma_uniform', 0.0001, 8)]),
                    'coef0': hp.uniform("svc_sigmoid_coef0", 0, 10)
                },
                {
                    'name': 'ploy',
                    'gamma': hp.choice('svc_ploy_gamma', ['auto', hp.uniform('svc_ploy_gamma_uniform', 0.0001, 8)]),
                    'coef0': hp.uniform("svc_ploy_coef0", 0, 10),
                    'degree': hp.uniformint("svc_ploy_degree", 1, 5),
                }
            ])
        }
    ]
)