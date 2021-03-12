from argparse import ArgumentParser

from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

from frame import Frame
from sklearn_pipeline import RegressionModel

parser = ArgumentParser(description='Pipeline to test sklearn models.')
parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs.')
args = parser.parse_args()

models = {'ExtraTreesRegressor': ExtraTreesRegressor(),
          'MLPRegressor': MLPRegressor(max_iter=5000, validation_fraction=0.2),
          'Ridge': Ridge()}

# Note that for random forests the 'mae' criterion is very slow
grid = {'Ridge': {'alpha': [0.0, 0.1, 0.5, 1.0]},
        'ExtraTreesRegressor': {'n_estimators': [256, 512, 1024],
                                'max_depth': [16, 32, 64, None],
                                'min_samples_split': [32, 64, 128, 256],
                                'min_impurity_decrease': [0.0, 0.1],
                                'max_leaf_nodes': [None]},
        'MLPRegressor': {'hidden_layer_sizes': [(8,), (16,),
                                                (16, 8, 4),
                                                (32, 16, 8),
                                                (32, 16, 8, 4),
                                                (64, 32, 16),
                                                (64, 32, 16, 8),
                                                (128, 64, 32, 16, 8),
                                                (256, 128, 64, 32, 16)],
                         'activation': ['tanh'],
                         'solver': ['adam'],
                         'batch_size': [128, 'auto'],
                         'alpha': [0.001, 0.01],
                         'learning_rate': ['adaptive'],
                         'early_stopping': [False]}}

comb_features = {'Mvir': ['log10', 'standard'],
                 'rho0': ['log10', 'standard'],
                 'spin': ['log10', 'standard'],
                 'potential': ['standard']}

# halo_features = {'Mvir': ['log10', 'standard'],
#                  'concentration': ['log10', 'standard'],
#                  'gamma': ['standard'],
#                  'spin': ['log10', 'standard'],
#                  'potential': ['standard']}

halo_features = {'Mvir': ['log10', 'standard'],
                 'concentration': ['log10', 'standard'],
                 'gamma': ['standard'],
                 'spin': ['log10', 'standard'],
                 'potential': ['standard'],
                 'rvir': ['log10', 'standard'],
                 'rho0': ['log10', 'standard'],
                 'L': ['log10', 'standard'],
                 'cvel': ['log10', 'standard'],
                 'Eint': ['standard'],
                 'Epot': ['log10', 'standard'],
                 'Ekin': ['log10', 'standard'],
                 'rs': ['log10', 'standard']}

# halo_features = {'Mvir': ['log10', 'standard']}

# halo_features = {'Mvir': ['log10', 'standard'],
#                  'concentration': ['log10', 'standard']}

# subhalo_features = {'Mvir': ['log10', 'standard']}

# subhalo_features = {'Mvir': ['log10', 'standard'],
#                     'Eint': ['standard'],
#                     'rs': ['log10', 'standard'],
#                     'rho0': ['log10', 'standard'],
#                     'gamma': ['standard'],
#                     'spin': ['log10', 'standard'],
#                     'parent_Mvir': ['log10', 'standard'],
#                     'potential': ['standard']}

# subhalo_features = {'Mvir': ['log10', 'standard'],
#                     'concentration': ['log10', 'standard']}

# subhalo_features = {'Mvir': ['log10', 'standard'],
#                     'concentration': ['log10', 'standard'],
#                     'gamma': ['standard'],
#                     'spin': ['log10', 'standard'],
#                     'potential': ['standard']}

subhalo_features = {'gamma': ['standard'],
                    'spin': ['log10', 'standard'],
                    'concentration': ['log10', 'standard'],
                    'potential': ['standard'],
                    'rho0': ['log10', 'standard']}

# subhalo_features = {'Mvir': ['log10', 'standard'],
#                     'concentration': ['log10', 'standard'],
#                     'gamma': ['standard'],
#                     'spin': ['log10', 'standard'],
#                     'potential': ['standard'],
#                     'rho0': ['log10', 'standard'],
#                     'Eint': ['standard'],
#                     'parent_Mvir': ['log10', 'standard'],
#                     'parent_L': ['log10', 'standard'],
#                     'L': ['log10', 'standard'],
#                     'Ekin': ['log10', 'standard'],
#                     'cvel': ['log10', 'standard'],
#                     'Epot': ['log10', 'standard'],
#                     'rs': ['log10', 'standard']}

target = {'label': 'Reff', 'transforms': ['log10', 'standard']}
search = 'grid'
subhalos = 2
n_iters = 10
perm_repeats = 100
seed = 42

if subhalos == 0:
    features = comb_features
elif subhalos == 1:
    features = halo_features
elif subhalos == 2:
    features = subhalo_features


for name in ['MLPRegressor', 'ExtraTreesRegressor']:
    # Define the frame
    frame = Frame(features, target, subhalos, seed=seed)

    model = RegressionModel(models[name], name, grid[name], frame,
                            search=search, n_iters=n_iters, n_jobs=args.n_jobs,
                            seed=seed, perm_repeats=perm_repeats, verbose=2)
    model.evaluate()
