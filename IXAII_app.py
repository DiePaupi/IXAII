# Note: Open http://127.0.0.1:8050/

# --- IMPORTS ------------------------------------------------------------------------------------------
from dash import Dash, html, Input, State, Output, dcc, ctx
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import math
import time
import plotly.express as px

from sklearn import datasets, model_selection

import IXAII_functions
import shap

# NOTE: Those are important to show plt plots (eg SHAP) in dash:
# https://github.com/plotly/tutorial-code/blob/main/Videos/matplotlib-dashboard.py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO




#############################################################################################################
#  Initial Python Stuff
#############################################################################################################

ML_METHOD = 'XGBReg'

DEFAULT_DISPLAY_COUNT = 4
DEFAULT_ANCHOR_THRESHOLD = 0.95
DEFAULT_nWHY_SHAP_CUTOFF = 0.01

DEFAULT_DICE_EXAMPLE_COUNT = 3
DEFAULT_DICE_MAX_EXAMPLE_COUNT = 5
DEFAULT_DICE_TRESHOLD = 0.5 # -> Must lie between 0.0 and 1.0

DEAFUALT_WHEN_TARGET_CLASS = 0
current_input_vals = []

# DATASET SPECIFIC -------------------------------------------------------------------------------------
# -> See https://www.kaggle.com/datasets/uciml/iris/data
pre_target_str = "Species "
# ['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
DEFAULT_DICE_FEATURES_TO_VARY = [1, 2, 3] 
# [ [f0_min, f0_max], [f1_min, f1_max], ... ]
feature_ranges = [ [3.0, 9.0], [1.0, 6.0], [0.0, 8.0], [0.0, 4.0]]



# 1) Load Data =========================================================================================
dataset = datasets.load_iris()
feature_list = dataset.feature_names
feature_count = len(feature_list)
target_names = dataset.target_names

# 1.1) Create a pandas DataFrame from the dataset ------------------------------------------------------
dataset_df, X_df = IXAII_functions.get_data_df(dataset, feature_list)


# 2) Get list of all outcomes ==========================================================================
target_list, outcome_tabel_array = IXAII_functions.get_outcomes(target_names)

dataset_df, X_df = IXAII_functions.get_data_df(dataset, feature_list)


# 3) Get prototypical inputs ==========================================================================
(data_row_count, data_col_count) = dataset_df.shape
avg_feauture_values = IXAII_functions.get_avg_input_set(dataset_df, data_row_count)
outcome_prototype_table = IXAII_functions.get_prototypical_input_sets(dataset_df, target_names, feature_list)

# 4) Train model and get predict =======================================================================
X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.data, dataset.target,
                                                                    test_size=0.1, random_state=0)

# NOTE: Using XGBClassifier does not work in combination with SHAP here!
model = IXAII_functions.get_model(ML_METHOD)
model.fit(X_train, y_train)

y_pred = model.predict(avg_feauture_values)
#pred_string = pre_target_str + str(y_pred[0])
pred_string = pre_target_str + target_list[round(y_pred[0])]


# 5) EXPLAINER =========================================================================================
shap_explainer = IXAII_functions.get_shap_explainer(ML_METHOD, model, X_train)

lime_explainer = IXAII_functions.get_lime_explainer(X_train, feature_list, target_names)

anchor_explainer = IXAII_functions.get_achnor_explainer(X_df.to_numpy(), feature_list, target_names)

dice_explainer = IXAII_functions.get_dice_explainer(ML_METHOD, model, dataset_df, feature_list)


# 4.1) Get global SHAP importance values ---------------------------------------------------------------
global_shap_values, global_shap_importance_df = IXAII_functions.get_global_shap_values(shap_explainer, X_df,
                                                                             X_train, feature_list)


# 4.2) Add LIME weights for global representation ------------------------------------------------------
lime_importance_df = IXAII_functions.get_lime_values(lime_explainer, model, X_train, feature_list)


# 4.3) Add both global values into a Plotly plot -------------------------------------------------------
global_importance_figure = IXAII_functions.get_global_importance_plot(global_shap_importance_df, lime_importance_df)

# -> Modify the tickangle of the xaxis, resulting in rotated labels
global_importance_figure.update_layout(barmode='group', legend_title_text = "Average |Values|",
                                       xaxis_tickangle=-45)


# 4.4) Get the Anchors prediction ----------------------------------------------------------------------
# TODO: Not sure if this works with MLPReg model
anchor_explanation = anchor_explainer.explain_instance(np.array(avg_feauture_values),
                                                model.predict, threshold=DEFAULT_ANCHOR_THRESHOLD)
anchor_rules = anchor_explanation.names()
rule_table_list = IXAII_functions.get_anchor_html_table_list(anchor_rules)
anchor_table = dbc.Table( [ html.Tbody(rule_table_list, id='anchor_rule_tablebody_display') ],
                          bordered=False)


# 4.5) Prepare DiCE feature variation input checkboxes -------------------------------------------
dice_feature_variation_checkboxes = IXAII_functions.get_dice_varying_features_checkboxes(feature_list)





#############################################################################################################
#####  #############    ########  ####  ######     #######  ####  ####         #################   ##########
#####  ############  ##  ########  ##  #####  #####  #####  ####  #######  ###################  #  ##########
#####  ###########  ####  ########    ######  #####  #####  ####  #######  #################  ###  ##########
#####  ##########  #    #  ########  #######  #####  #####  ####  #######  ######################  ##########
#####       ####  ########  #######  #########     #######        #######  ######################  ##########
#############################################################################################################

#############################################################################################################
#  Get the Explanation Tabs' Layout
#############################################################################################################

# --- Starting Tab -------------------------------------------------------------------------------------
starting_tab = dbc.Tab(
                    dbc.Card( dbc.CardBody([
                        html.H3("Welcome to the Interactive XAI interface (IXAII)!",
                                className='card-text'),
                        html.H4("1) Please type the input values into the left column",
                                className='card-text'),
                        html.H4("2) Klick on the 'Generate Prediction' button to update the "
                                 + "prediction according to your input",
                                className='card-text'),
                        html.H4("3) Choose a user profile through the drop down menu in the upper "
                                 + "right corner to see relevant explantions for your prediction",
                                className='card-text')
                    ]), className='mt-3'),
                label="Get started!", labelClassName='text-primary',
                tab_style={'font-weight': 'bold'})


# --- Data Exploration Tab -----------------------------------------------------------------------------
data_exp_tab = dbc.Tab([
                    html.P("The Data Exploration provides information about the training data.",
                        className='mt-3'),
                    dbc.Tabs([
                        # --- Info TAB ------------------------------------------------------------
                        dbc.Tab([
                            dbc.Card( dbc.CardBody([
                                html.H4("These are all current inputs:",
                                    id='de_inputs_overview_title'),
                                dbc.Tooltip("This list presents all considered input values which "
                                             + "may also include sensor data that cannot be manually "
                                             + "adjusted.",
                                    target='de_inputs_overview_title'),
                                dbc.Table(id='de_inputs_overview_out', bordered=False,
                                    className='mt-4 table-hover'),

                                html.H4("These are all possible outcomes:",
                                    id='de_outcome_overview_title'),
                                dbc.Tooltip("Here, the output classes represent the three different "
                                             + "cultivars the wine data was taken from.",
                                    target='de_outcome_overview_title'),
                                dbc.Table(outcome_tabel_array, id='de_outcome_overview_out',
                                    bordered=False, className='mt-4 table-hover'),

                                html.H4("These are prototypical inputs for each outcome:",
                                    id='de_prototypical_inputs_overview_title'),
                                dbc.Tooltip("The presented values represent the average values "
                                             + "of the training data for each outcome's feature.",
                                    target='de_prototypical_inputs_overview_title'),
                                dbc.Table(outcome_prototype_table, id='de_prototypical_inputs_overview_out',
                                    bordered=False, className='mt-4 table-hover')
                            ]), className='mt-3'),
                        ], label="Information", labelClassName='text-primary'),
                        # --- Histogram TAB -------------------------------------------------------
                        dbc.Tab([
                            dbc.Card( dbc.CardBody([
                                dcc.Graph(figure={}, id='histogram'),
                                dbc.InputGroup([
                                    dbc.InputGroupText("Select a feature for the histogram's y axis:"),
                                    dbc.Select(options=feature_list,
                                        id='histogram_feature_dropdown', value='sepal length (cm)')
                                ])
                            ]), className='mt-3'),
                        ], label="Histogram", labelClassName='text-primary'),
                        # --- Boxplot TAB ---------------------------------------------------------
                        dbc.Tab([
                            dbc.Card( dbc.CardBody([
                                dcc.Graph(figure={}, id='de_boxplot_graph'),
                                dbc.InputGroup([
                                    dbc.InputGroupText("Select a feature for the box plot's y axis:"),
                                    dbc.Select(options=feature_list, id='de_boxplot_y_dropdown',
                                        value='sepal length (cm)')
                                ]),

                                dbc.InputGroup([
                                    dbc.InputGroupText("Display underlaying data points:"),
                                    dbc.InputGroupText(dbc.Checkbox(id='de_boxplot_display_points'))
                                ], className='mt-3')
                            ]), className='mt-3'),
                        ], label="Box Plot", labelClassName='text-primary'),
                        # --- Scatterplot TAB --------------------------------------
                        dbc.Tab([
                            dbc.Card( dbc.CardBody([
                                dcc.Graph(figure={}, id='de_scatterplot_graph'),
                                dbc.InputGroup([
                                    dbc.InputGroupText("Select the y and x features:"),
                                    dbc.Select(options=feature_list, id='de_scatter_y_dropdown',
                                        value='sepal length (cm)'),
                                    dbc.Select(options=feature_list, id='de_scatter_x_dropdown',
                                        value='sepal width (cm)')
                                ])
                            ]), className='mt-3'),
                        ], label="Scatter Plot", labelClassName='text-primary')
                    ], className='mt-3')
                ],
                label="Data Exploration", id='data_exploration_tab_label',
                labelClassName='text-primary', tab_style={'font-weight': 'bold'})


# --- Why Explanation Tab ------------------------------------------------------------------------------
why_exp_tab = dbc.Tab([
                    html.P("Why Explanations give information about why the system derived the "
                            + "current output from the given inputs.",
                        className='mt-3'),
                    dbc.Tabs([
                        # --- Global LIME TAB -----------------------------------------------------
                        dbc.Tab([
                            dbc.Card( dbc.CardBody([
                                html.H4("Group comparison between LIME's and SHAP's global values",
                                    id='why_global_plottype_title'),
                                dcc.Graph(figure=global_importance_figure,
                                          id='why_global_plottype_figure'),
                                dbc.InputGroup([
                                    dbc.InputGroupText("Select the plot type:"),
                                    dbc.Select(options=['Comparison: Group', 'Comparison: Stack',
                                                        'Single: LIME', 'Single: SHAP'], 
                                        id='why_global_plottype_dropdown', value='Comparison: Group')
                                ], className='mt-3'),
                            ]), className='mt-3')
                        ], label="Global", labelClassName='text-primary'),
                        # --- Local LIME TAB ------------------------------------------------------
                        dbc.Tab([
                            dbc.Card( dbc.CardBody([
                                html.Div(id='why_local_lime_figure', style={'text-align': 'center',
                                                                            'width':'100%'}),
                                dbc.InputGroup([
                                    dbc.InputGroupText("Select the number of features to include:",
                                        id='why_local_lime_featureno_label'),
                                    dbc.Input(id='why_local_lime_featureno_input', 
                                        type='number', step='1', min='1', max=feature_count)
                                ], className='mt-3'),
                                dbc.Tooltip("LIME will consider this many of the most important "
                                             + "features. Accordingly, the values of the calculated "
                                             + "weights change when this number is adjusted.",
                                    target='why_local_lime_featureno_label')
                            ]), className='mt-3')
                        ], label="Local LIME", labelClassName='text-primary'),
                        # --- SHAP TAB ------------------------------------------------------------
                        dbc.Tab([
                            dbc.Card( dbc.CardBody([
                                #html.Img(id='why_shap_figure', style={'width':'90%'}),
                                html.Div(id='why_shap_figure'),
                                dbc.InputGroup([
                                    dbc.InputGroupText("Select the SHAP plot type:"),
                                    dbc.Select(options=['local_bar', 'waterfall'], 
                                        id='why_shap_plottype_dropdown', value='local_bar')
                                ], className='mt-3'),
                                dbc.InputGroup([
                                    dbc.InputGroupText("Select the number of features to include:",
                                        id='why_shap_featureno_label'),
                                    dbc.Input(id='why_shap_featureno_input', 
                                        placeholder=str(DEFAULT_DISPLAY_COUNT), type='number',
                                        step='1', min='1', max=feature_count)
                                ], className='mt-3'),
                                dbc.Tooltip("This graph will present this number of the most "
                                             + "important features. "
                                             + "In contrast to LIME, this will not change the "
                                             + "features' weights, just the extend of the graph.",
                                    target='why_shap_featureno_label')
                            ]), className='mt-3')
                        ], label="Local SHAP", labelClassName='text-primary'),
                        # --- Anchor --------------------------------------------------------------
                        dbc.Tab([
                            dbc.Card( dbc.CardBody([

                                # --- format method -------------------------------------
                                html.Div([
                                    dbc.Label("Select a format method:",
                                        id='anchor_why_format_method_label'),
                                    dbc.RadioItems(
                                        options=[{'label': "Table", 'value': 0},
                                                {'label': "Bar Plot", 'value': 1}],
                                        value=0,
                                        id="anchor_why_format_method_input",
                                        inline=True,
                                    ),
                                ], className='mt-3'),
                                dbc.Tooltip("Choose if you want the examples displayed as a table or "
                                            "as a box plot.",
                                    target='anchor_why_format_method_label'),

                                # --- treshld -------------------------------------------
                                dbc.InputGroup([
                                    dbc.InputGroupText("Select Anchor's threshold:",
                                        id='anchor_threshold_label'),
                                    dbc.Input(id='anchor_threshold_input',
                                              placeholder=str(DEFAULT_ANCHOR_THRESHOLD),
                                        type='number', step='0.05', min='0.05', max='1.00')
                                ], className='mt-3 mb-3'),
                                dbc.Tooltip("This threshold x determines the precision of Anchor, "
                                             + "meaning that predictions on instances where the "
                                             + "anchor holds will be the same as the original "
                                             + "prediction at least x percent of the time.",
                                    target='anchor_threshold_label'),
                                    
                                # --- output --------------------------------------------
                                html.Div(anchor_table, id='anchor_why_out')
                            ]), className='mt-3')
                        ], label="Anchor", labelClassName='text-primary'),
                        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    ], className='mt-3')
                ],
                label="Why", id='why_explanation_tab_label',
                labelClassName='text-primary', tab_style={'font-weight': 'bold'})


# --- nWhy Explanation Tab -----------------------------------------------------------------------------
nWhy_exp_tab = dbc.Tab([
                    html.P("-Why Explanations give Information about which inputs were not relevant "
                            + "to the current output.",
                        className='mt-3'),
                    dbc.Card( dbc.CardBody([
                        dbc.InputGroup([
                            dbc.InputGroupText("All below listed values have a mean SHAP value "
                                                + "smaller than (or equal to):",
                                               id='DEFAULT_nWHY_SHAP_CUTOFF_label'),
                            dbc.Input(id='DEFAULT_nWHY_SHAP_CUTOFF_input',
                                      placeholder=str(DEFAULT_nWHY_SHAP_CUTOFF),
                                type='number', step='0.01', min='0.0', max='1.0')
                        ], className='mt-3 mb-3'),
                        dbc.Tooltip("This list is based on the SHAP values of the selected plottype "
                                     + "below.",
                            target='DEFAULT_nWHY_SHAP_CUTOFF_label'),
                        html.Ol(id='nWhy_shap_out'),

                        html.Hr(),

                        html.Img(id='nWhy_shap_figure', style={'width':'90%'}),
                        dbc.InputGroup([
                            dbc.InputGroupText("Select the SHAP plot type:"),
                            dbc.Select(options=['global_bar', 'local_bar'],
                                       id='nWhy_shap_plottype_dropdown',
                                value='global_bar')
                        ]),
                        dbc.InputGroup([
                            dbc.InputGroupText("Select the number of features to include:",
                                id='nWhy_shap_featureno_label'),
                            dbc.Input(id='nWhy_shap_featureno_input',
                                placeholder=str(DEFAULT_DISPLAY_COUNT), type='number',
                                step='1', min='1', max=feature_count)
                        ], className='mt-3'),
                        dbc.Tooltip("This graph will present this number of the least important "
                                     + "features. "
                                     + "In contrast to LIME, this will not change the features' "
                                     + "weights, just the extend of the graph.",
                            target='nWhy_shap_featureno_label')
                    ]), className='mt-3')
                ],
                label="-Why", id='nWhy_explanation_tab_label',
                labelClassName='text-primary', tab_style={'font-weight': 'bold'})


# --- Why Not Explanation Tab --------------------------------------------------------------------------
why_not_exp_tab = dbc.Tab([
                    html.P("Why Not Explanations give information about why another possible outcome "
                            + "was not derived w.r.t. derived outcome and the applied inputs.",
                        className='mt-3'),
                    dbc.Card( dbc.CardBody([
                        html.H4("Counterfactual examples with DiCE:",
                            id='why_not_heading_label'),
                        dbc.Tooltip("Counterfactual examples specifically have another outcome so "
                                     + "you can compare the inputs to your original values.",
                            target='why_not_heading_label'),

                        # --- example no ----------------------------------------------------------
                        dbc.InputGroup([
                            dbc.InputGroupText("Number of counterfactuals to generate:",
                                id='dice_why_not_no_label'),
                            dbc.Input(id='dice_why_not_no_input',
                                      placeholder=str(DEFAULT_DICE_EXAMPLE_COUNT),
                                type='number', step='1', min='1', max=DEFAULT_DICE_MAX_EXAMPLE_COUNT)
                        ], className='mt-3'),
                        dbc.Tooltip("Must be an integer 0 < x <= 5",
                            target='dice_why_not_no_label'),

                        # --- treshold ------------------------------------------------------------
                        dbc.InputGroup([
                            dbc.InputGroupText("Define DiCE's threshold:",
                                id="dice_why_not_threshold_label"),
                            dbc.Input(id='dice_why_not_threshold_input',
                                      placeholder=str(DEFAULT_DICE_TRESHOLD),
                                type='number', step='0.1', min='0.0', max='1.0')
                        ], className='mt-3'),
                        dbc.Tooltip("Minimum threshold for counterfactuals target class probability. "
                                    "Must be a value between 0.0 and 1.0",
                            target='dice_why_not_threshold_label'),

                        # --- features to vary ----------------------------------------------------
                        html.Div([
                            dbc.Label("Select the features to vary:",
                                id='dice_why_not_features_to_vary_label'),
                            dbc.Checklist(
                                options=dice_feature_variation_checkboxes,
                                value=DEFAULT_DICE_FEATURES_TO_VARY,
                                id="dice_why_not_features_to_vary_input",
                                inline=True,
                            ),
                        ], className='mt-3'),
                        dbc.Tooltip("The features for which different input values (from your "
                                    "original input) should be considered. "
                                    "Please select at least one.",
                            target='dice_why_not_features_to_vary_label'),

                        # --- format method -------------------------------------------------------
                        html.Div([
                            dbc.Label("Select a format method:",
                                id='dice_why_not_format_method_label'),
                            dbc.RadioItems(
                                options=[{'label': "Table", 'value': 0},
                                         {'label': "Bar Plot", 'value': 1}],
                                value=0,
                                id="dice_why_not_format_method_input",
                                inline=True,
                            ),
                        ], className='mt-3'),
                        dbc.Tooltip("Choose if you want the examples displayed as a table or "
                                    "as a bar plot.",
                            target='dice_why_not_format_method_label'),

                        # --- generate ------------------------------------------------------------
                        dbc.Button("Generate Examples", id='dice_why_not_generation_button',
                                   type='submit', n_clicks=0,
                            outline=True, color='primary', size='sm', className='mt-3 me-1'),

                        # --- output --------------------------------------------------------------
                        html.Div(id='dice_why_not_out', className='mt-3')

                    ]), className='mt-3')
                ],
                label="Why Not", id='why_not_explanation_tab_label',
                labelClassName='text-primary', tab_style={'font-weight': 'bold'})


# --- What If Explanation Tab --------------------------------------------------------------------------
what_if_exp_tab = dbc.Tab([
                    html.P("What If Explanations display simulated outputs based on altered inputs.",
                        className='mt-3'),
                    dbc.Card( dbc.CardBody([
                        html.H4("Factual examples with DiCE:",
                            id='what_if_heading_label'),
                        dbc.Tooltip("Factual examples have similar inputs to your original values "
                                    "so you can compare the outcomes.",
                            target='what_if_heading_label'),

                        dbc.InputGroup([
                            dbc.InputGroupText("Number of factual examples to generate:",
                                id='dice_what_if_no_label'),
                            dbc.Input(id='dice_what_if_no_input',
                                      placeholder=str(DEFAULT_DICE_EXAMPLE_COUNT),
                                type='number', step='1', min='1', max=DEFAULT_DICE_MAX_EXAMPLE_COUNT)
                        ], className='mt-3'),
                        dbc.Tooltip("Must be an integer 0 < x <= 5",
                            target='dice_what_if_no_label'),

                        dbc.InputGroup([
                            dbc.InputGroupText("Define DiCE's threshold:",
                                id="dice_what_if_threshold_label"),
                            dbc.Input(id='dice_what_if_threshold_input',
                                      placeholder=str(DEFAULT_DICE_TRESHOLD),
                                type='number', step='0.1', min='0.0', max='1.0')
                        ], className='mt-3'),
                        dbc.Tooltip("Minimum threshold for counterfactuals target class probability. "
                                    "Must be a value between 0.0 and 1.0",
                            target='dice_what_if_threshold_label'),

                        html.Div([
                            dbc.Label("Select the features to vary:",
                                id='dice_what_if_features_to_vary_label'),
                            dbc.Checklist(
                                options=dice_feature_variation_checkboxes,
                                value=DEFAULT_DICE_FEATURES_TO_VARY,
                                id="dice_what_if_features_to_vary_input",
                                inline=True,
                            ),
                        ], className='mt-3'),
                        dbc.Tooltip("The features for which different input values (from your "
                                    "original input) should be considered. "
                                    "Please select at least one.",
                            target='dice_what_if_features_to_vary_label'),

                        html.Div([
                            dbc.Label("Select a format method:",
                                id='dice_what_if_format_method_label'),
                            dbc.RadioItems(
                                options=[{'label': "Table", 'value': 0},
                                         {'label': "Bar Plot", 'value': 1}],
                                value=0,
                                id="dice_what_if_format_method_input",
                                inline=True,
                            ),
                        ], className='mt-3'),
                        dbc.Tooltip("Choose if you want the examples displayed as a table or "
                                    "as a bar plot.",
                            target='dice_what_if_format_method_label'),

                        dbc.Button("Generate Examples", id='dice_what_if_generation_button',
                                   type='submit', n_clicks=0,
                            outline=True, color='primary', size='sm', className='mt-3 me-1'),

                        html.Div(id='dice_what_if_out', className='mt-3')
                    ]), className='mt-3')
                ],
                label="What If", id='what_if_explanation_tab_label',
                labelClassName='text-primary', tab_style={'font-weight': 'bold'})


# --- When Explanation Tab -----------------------------------------------------------------------------
when_exp_tab = dbc.Tab([
                    html.P("When Explanations display sumulated inputs based on a desired output.",
                        className='mt-3'),
                    dbc.Card( dbc.CardBody([
                        html.H4("Counter- and Semi-factual examples with DiCE:",
                            id='when_heading_label'),
                        dbc.Tooltip("Counterfactual / Semi-factual examples specifically have "
                                    "another / the same outcome so you can "
                                    "compare the inputs to your original values.",
                            target='when_heading_label'),

                        dbc.InputGroup([
                            dbc.InputGroupText("Target class:",
                                id='dice_when_target_class_label'),
                            dbc.Input(id='dice_when_target_class_input', type='number',
                                step='1', min='0', max=len(target_list)-1)
                        ], className='mt-3'),
                        dbc.Tooltip("Which outcome should the generated examples have? 0, 1, or 2?",
                            target='dice_when_target_class_label'),

                        dbc.InputGroup([
                            dbc.InputGroupText("Number of examples to generate:",
                                id='dice_when_no_label'),
                            dbc.Input(id='dice_when_no_input',
                                      placeholder=str(DEFAULT_DICE_EXAMPLE_COUNT),
                                type='number', step='1', min='1', max=DEFAULT_DICE_MAX_EXAMPLE_COUNT)
                        ], className='mt-3'),
                        dbc.Tooltip("Must be an integer 0 < x <= 5",
                            target='dice_when_no_label'),

                        dbc.InputGroup([
                            dbc.InputGroupText("Define DiCE's threshold:",
                                id="dice_when_threshold_label"),
                            dbc.Input(id='dice_when_threshold_input',
                                      placeholder=str(DEFAULT_DICE_TRESHOLD),
                                type='number', step='0.1', min='0.0', max='1.0')
                        ], className='mt-3'),
                        dbc.Tooltip("Minimum threshold for counterfactuals target class probability. "
                                    "Must be a value between 0.0 and 1.0",
                            target='dice_when_threshold_label'),

                        html.Div([
                            dbc.Label("Select the features to vary:",
                                id='dice_features_to_vary_label'),
                            dbc.Checklist(
                                options=dice_feature_variation_checkboxes,
                                value=DEFAULT_DICE_FEATURES_TO_VARY,
                                id="dice_when_features_to_vary_input",
                                inline=True,
                            ),
                        ], className='mt-3'),
                        dbc.Tooltip("The features for which different input values (from your "
                                    "original input) should be considered. "
                                    "Please select at least one.",
                            target='dice_features_to_vary_label'),

                        # --- format method -------------------------------------------------------
                        html.Div([
                            dbc.Label("Select a format method:",
                                id='dice_when_format_method_label'),
                            dbc.RadioItems(
                                options=[{'label': "Table", 'value': 0},
                                         {'label': "Bar Plot", 'value': 1}],
                                value=0,
                                id="dice_when_format_method_input",
                                inline=True,
                            ),
                        ], className='mt-3'),
                        dbc.Tooltip("Choose if you want the examples displayed as a table or "
                                    "as a bar plot.",
                            target='dice_when_format_method_label'),

                        dbc.Button("Generate Examples", id='dice_when_generation_button',
                                   type='submit', n_clicks=0,
                            outline=True, color='primary', size='sm', className='mt-3 me-1'),

                        html.Div(id='dice_when_out', className='mt-3')

                    ]), className='mt-3')
                ],
                label="When", id='when_explanation_tab_label',
                labelClassName='text-primary', tab_style={'font-weight': 'bold'})


# --- Settings Tab -------------------------------------------------------------------------------------
settings_tab = dbc.Tab(
                    dbc.Card( dbc.CardBody(
                        dbc.Form([
                            html.Div([
                                dbc.Label("Please select the explanation types to display:"),
                                dbc.Checklist(
                                    options=[
                                        {"label": "Data Exploration", "value": 1},
                                        {"label": "Why", "value": 2},
                                        {"label": "-Why", "value": 3},
                                        {"label": "Why Not", "value": 4},
                                        {"label": "What If", "value": 5},
                                        {"label": "When", "value": 6}
                                    ],
                                    value=[], id="explanations_toggle_input",
                                ),
                            ], style={'text-align': 'left'})
                        ])
                    ), className='mt-3'),
                    id='settings_tab', label="Settings", labelClassName='text-info',
                    tab_style={'marginLeft': 'auto', 'font-weight': 'bold'}
                )



#############################################################################################################
#  User Profile Definitions
#############################################################################################################

dev_user_profile_values = [1, 2, 3, 6] # Data Exploration, Why, nWhy, and When
dev_user_profile_list = [ data_exp_tab, why_exp_tab, nWhy_exp_tab, when_exp_tab, settings_tab ]

user_user_profile_vlaues = [2, 3, 4, 5]   # Why, nWhy, Why Not, and What If
user_user_profile_list = [ why_exp_tab, nWhy_exp_tab, why_not_exp_tab, what_if_exp_tab, settings_tab ]

business_user_profile_vlaues = [2, 4, 5, 6]  # Why, Why Not, What If, and When
business_user_profile_list = [ why_exp_tab, why_not_exp_tab, what_if_exp_tab, when_exp_tab, settings_tab ]

regulatory_user_profile_vlaues = [1, 2, 4, 6]  # Data Exploration, Why, Why Not, When
regulatory_user_profile_list = [ data_exp_tab, why_exp_tab, why_not_exp_tab, when_exp_tab, settings_tab ]

affected_user_profile_vlaues = [2, 4, 5]  # Why, Why Not, What If
affected_user_profile_list = [ why_exp_tab, why_not_exp_tab, what_if_exp_tab, settings_tab ]



#############################################################################################################
#  Start the Dash app
#############################################################################################################

# --- INIT ---------------------------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])





#############################################################################################################
#####  #############    ########  ####  ######     #######  ####  ####         ##############       #########
#####  ############  ##  ########  ##  #####  #####  #####  ####  #######  #################  #####  ########
#####  ###########  ####  ########    ######  #####  #####  ####  #######  ######################  ##########
#####  ##########  #    #  ########  #######  #####  #####  ####  #######  ####################  ############
#####       ####  ########  #######  #########     #######        #######  ##################        ########
#############################################################################################################

app.layout = dbc.Container([
    dbc.Container([dbc.NavbarSimple(
        children=[
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Developer", id="dropdown_up_dev_input", n_clicks=0),
                    dbc.DropdownMenuItem("User", id="dropdown_up_user_input", n_clicks=0),
                    dbc.DropdownMenuItem("Business Entity", id="dropdown_up_business_input", n_clicks=0),
                    dbc.DropdownMenuItem("Regulatory Entity", id="dropdown_up_regulatory_input", n_clicks=0),
                    dbc.DropdownMenuItem("Affected Party", id="dropdown_up_affected_input", n_clicks=0),
                ],
                nav=True,
                in_navbar=True,
                label="Select User Profile",
            ),
        ],
        brand="Interactive XAI Interface",
        color='primary', dark=True, fixed='top'
    )], className='container-fluid navbar-bottom-padding'),

    dbc.Row([

        # === INPUTS COL ===============================================================================
        dbc.Col([

            # - Inputs ----------------------------------------------------------------------------
            dbc.Row([
                html.H5("Inputs:")
            ], className='row bg-primary rounded-3 mb-3'),
            html.Div([
                dbc.InputGroup([
                    dbc.Input(id='sepal_length-input', placeholder=str(avg_feauture_values[0][0]),
                              type='number', step='0.1',
                              min=str(feature_ranges[0][0]), max=str(feature_ranges[0][1])),
                    dbc.InputGroupText("= Sepal Length (cm)", id='input_label_sepal_length'),
                    dbc.Tooltip(html.Div([
                            html.Div("Length of the sepal (in cm)"),
                            html.Div(" - "),
                            html.Div("Range: " + str(feature_ranges[0][0]) + "<= x <="
                                      + str(feature_ranges[0][1]))
                        ]),
                        target='input_label_sepal_length')
                ], size='sm', className='mb-2'),
                dbc.InputGroup([
                    dbc.Input(id='sepal_width-input', placeholder=str(avg_feauture_values[0][1]),
                              type='number', step='0.1',
                              min=str(feature_ranges[1][0]), max=str(feature_ranges[1][1])),
                    dbc.InputGroupText("= Sepal Width (cm)", id='input_label_sepal_width'),
                    dbc.Tooltip(html.Div([
                            html.Div("Width of the sepal (in cm)"),
                            html.Div(" - "),
                            html.Div("Range: " + str(feature_ranges[1][0]) + "<= x <="
                                      + str(feature_ranges[1][1]))
                        ]),
                        target='input_label_sepal_width')
                ], size='sm', className='mb-2'),
                dbc.InputGroup([
                    dbc.Input(id='petal_length-input', placeholder=str(avg_feauture_values[0][2]),
                              type='number', step='0.1',
                              min=str(feature_ranges[2][0]), max=str(feature_ranges[2][1])),
                    dbc.InputGroupText("= Petal Length (cm)", id='input_label_petal_length'),
                    dbc.Tooltip(html.Div([
                            html.Div("Length of the petal (in cm)"),
                            html.Div(" - "),
                            html.Div("Range: " + str(feature_ranges[2][0]) + "<= x <="
                                      + str(feature_ranges[2][1]))
                        ]),
                        target='input_label_petal_length')
                ], size='sm', className='mb-2'),
                dbc.InputGroup([
                    dbc.Input(id='petal_width-input', placeholder=str(avg_feauture_values[0][3]),
                              type='number', step='0.1',
                              min=str(feature_ranges[3][0]), max=str(feature_ranges[3][1])),
                    dbc.InputGroupText("= Petal Width (cm)", id='input_label_petal_width'),
                    dbc.Tooltip(html.Div([
                            html.Div("Width of the petal (in cm)"),
                            html.Div(" - "),
                            html.Div("Range: " + str(feature_ranges[3][0]) + "<= x <="
                                      + str(feature_ranges[3][1]))
                        ]),
                        target='input_label_petal_width')
                ], size='sm', className='mb-2'),
            ]),

            # - Predict Button --------------------------------------------------------------------
            dbc.Row([
                dbc.Col([
                    dbc.Button("Generate Prediction", id='predict_button', type='submit', n_clicks=0,
                               outline=True, color='primary', size='sm', className='me-1')
                ], width={'size': 8, 'offset': 2})
            ]),

            # - Prediction ------------------------------------------------------------------------
            dbc.Row([
                html.H5("Prediction:")
            ], className='row bg-primary rounded-3 mt-4 mb-3'),
            dbc.Row([
                html.Span(id="prediction_output", style={"verticalAlign": "middle"})
            ])
        ], width={"size": 3, "order": 1, "offset": 0}),

        dbc.Col([
            html.Div([
                dbc.Tabs([ starting_tab, settings_tab ], id='explanation_tabs')
            ])
        ], width={"size": 8, "order": 2, "offset": 1})
    ])
], className='container-fluid')





#############################################################################################################
###       ########     ########  #########  ##########      #########     ########       ####  ###  #########
###  ############  ###  #######  #########  ##########  ###  #######  ###  #######  #########  ##  ##########
###  ###########  #####  ######  #########  ##########  ##  #######  #####  ######  #########     ###########
###  ##########  ##   ##  #####  #########  ##########  ###  #####  ##   ##  #####  #########  ##  ##########
###       ####  #########  ####       ####        ####     ######  #########  ####       ####  ###  #########
#############################################################################################################

# --- Switch UP ----------------------------------------------------------------------------------------
@app.callback(
    Output('explanation_tabs', 'children'),
    Output('explanations_toggle_input', 'value'),
    [
        Input('dropdown_up_dev_input', 'n_clicks'),
        Input('dropdown_up_user_input', 'n_clicks'),
        Input('dropdown_up_business_input', 'n_clicks'),
        Input('dropdown_up_regulatory_input', 'n_clicks'),
        Input('dropdown_up_affected_input', 'n_clicks'),
        Input('explanations_toggle_input', 'value')
    ],
)
def switch_to_dev_user_profile(dev_nc, user_nc, business_nc, regulatory_nc, affected_nc, exp_toggle_values):
    profile_id = ctx.triggered_id
    match profile_id:
        case 'dropdown_up_dev_input':
            return dev_user_profile_list, dev_user_profile_values
        case 'dropdown_up_user_input':
            return user_user_profile_list, user_user_profile_vlaues
        case 'dropdown_up_business_input':
            return business_user_profile_list, business_user_profile_vlaues
        case 'dropdown_up_regulatory_input':
            return regulatory_user_profile_list, regulatory_user_profile_vlaues
        case 'dropdown_up_affected_input':
            return affected_user_profile_list, affected_user_profile_vlaues
        case 'explanations_toggle_input':
            new_exp_list = []
            new_exp_val_list = []
            if 1 in exp_toggle_values:
                new_exp_list.append(data_exp_tab)
                new_exp_val_list.append(1)
            if 2 in exp_toggle_values:
                new_exp_list.append(why_exp_tab)
                new_exp_val_list.append(2)
            if 3 in exp_toggle_values:
                new_exp_list.append(nWhy_exp_tab)
                new_exp_val_list.append(3)
            if 4 in exp_toggle_values:
                new_exp_list.append(why_not_exp_tab)
                new_exp_val_list.append(4)
            if 5 in exp_toggle_values:
                new_exp_list.append(what_if_exp_tab)
                new_exp_val_list.append(5)
            if 6 in exp_toggle_values:
                new_exp_list.append(when_exp_tab)
                new_exp_val_list.append(6)
            new_exp_list.append(settings_tab)
            return new_exp_list, new_exp_val_list
        case _:
            return [starting_tab, settings_tab], []


# --- Predict --------------------------------------------------------------------------------
@app.callback(
    Output('prediction_output', 'children', allow_duplicate=True),
    Input('predict_button', 'n_clicks'),
    prevent_initial_call=True
)
def update_loading_animation(n_clicks):
    return dbc.Spinner(color="primary", size="sm")

@app.callback(
    Output('prediction_output', 'children'),
    [
        Input('predict_button', 'n_clicks'),
        State('sepal_length-input', 'value'), 
        State('sepal_width-input', 'value'),
        State('petal_length-input', 'value'),
        State('petal_width-input', 'value')
    ]
)
def update_prediction(n_clicks, sepal_length, sepal_width, petal_length, petal_width):
    global current_input_vals 
    current_input_vals = [[
        float(sepal_length)     if sepal_length != None     else avg_feauture_values[0][0], 
        float(sepal_width)      if sepal_width != None      else avg_feauture_values[0][1], 
        float(petal_length)     if petal_length != None     else avg_feauture_values[0][2], 
        float(petal_width)      if petal_width != None      else avg_feauture_values[0][3]
    ]]
    print("Updating input values:")
    print("sepal_length =", sepal_length, "sepal_width =", sepal_width,
          "petal_length =", petal_length, "petal_width =", petal_width)

    global y_pred
    y_pred = model.predict(current_input_vals)

    global pred_string
    pred_string = pre_target_str + target_list[round(y_pred[0])]

    # Time whre the loading spinner is displayed (see update_loading_animation())
    time.sleep(1)

    return pred_string


# --- Input Overview ------------------------------------------------------------------------------
@app.callback(
    Output(component_id='de_inputs_overview_out', component_property='children'),
    Input('predict_button', 'n_clicks')
)
def update_input_overview(n_clicks):
    table_header = [html.Thead(html.Tr([
        html.Th("Feature"), html.Th("Range"), html.Th("Input Value")
    ], className='table-primary'))]
    
    table_body = []
    for i in range(len(feature_list)):
        feature_range = "[" + str(feature_ranges[i][0]) + " ; " + str(feature_ranges[i][1]) + "]"
        row = html.Tr([
            html.Td(feature_list[i]), html.Td(feature_range), html.Td(str(current_input_vals[0][i]))
        ])
        table_body.append(row)
        
    return dbc.Table(table_header + [html.Tbody(table_body)])


# --- Histogram -----------------------------------------------------------------------------------
@app.callback(
    Output(component_id='histogram', component_property='figure'),
    [   Input('predict_button', 'n_clicks'),
        Input(component_id='histogram_feature_dropdown', component_property='value')
    ])
def update_histogram(n_clicks, feature):
    return px.histogram(dataset_df, x='target_names', y=feature, histfunc='avg')


# --- Box Plot ------------------------------------------------------------------------------------
@app.callback(
    Output(component_id='de_boxplot_graph', component_property='figure'),
    [   Input('predict_button', 'n_clicks'),
        Input(component_id='de_boxplot_y_dropdown', component_property='value'),
        Input(component_id='de_boxplot_display_points', component_property='value')
    ])
def update_box_plot(n_clicks, feature, display_points):
    if display_points:
        return px.box(dataset_df, x='target_names', y=feature, points="all")
    else:
        return px.box(dataset_df, x='target_names', y=feature)


# --- Scatter Plot --------------------------------------------------------------------------------
@app.callback(
    Output(component_id='de_scatterplot_graph', component_property='figure'), 
    [   Input('predict_button', 'n_clicks'),
        Input(component_id='de_scatter_y_dropdown', component_property='value'),
        Input(component_id='de_scatter_x_dropdown', component_property='value'),
    ])
def update_de_scatterplot(n_clicks, y_feature, x_feature):
    return px.scatter(dataset_df, y=y_feature, x=x_feature, color='target_names')


# --- Why Global Plot -----------------------------------------------------------------------------
@app.callback(
    Output(component_id='why_global_plottype_title', component_property='children'), 
    Output(component_id='why_global_plottype_figure', component_property='figure'), 
    Input(component_id='why_global_plottype_dropdown', component_property='value'))
def update_de_global_feature_importance_plot(plottype):
    match plottype:
        case 'Comparison: Group':
            global_importance_figure.update_layout(barmode='group',
                legend_title_text = "Average |Values|", xaxis_tickangle=-45)
            return "Group comparison between LIME's and SHAP's global values", global_importance_figure
        case 'Comparison: Stack':
            global_importance_figure.update_layout(barmode='stack',
                legend_title_text = "Average |Values|", xaxis_tickangle=-45)
            return "Stack comparison between LIME's and SHAP's global values", global_importance_figure
        case 'Single: LIME':
            df_lime_importance_plot = px.bar(lime_importance_df, x='abs_mean', y='feature',
                orientation='h', labels={'abs_mean': "Mean |Weight|", 'feature': ""})
            return "Overview of LIME's global mean absolute wights", df_lime_importance_plot
        case 'Single: SHAP':
            shap_global_importance_plot = px.bar(global_shap_importance_df, x='importance', y='feature',
                orientation='h', labels={'importance': "Mean |SHAP vlaue|", 'feature': ""})
            return "Overview of SHAP's global mean absolute values", shap_global_importance_plot


# --- Why SHAP Plots ------------------------------------------------------------------------------
@app.callback(
    Output(component_id='why_shap_figure', component_property='children'), 
    [   Input('predict_button', 'n_clicks'),
        Input(component_id='why_shap_plottype_dropdown', component_property='value'),
        Input(component_id='why_shap_featureno_input', component_property='value')
    ])
def update_why_shapplot(n_clicks, plottype, feature_no_int):
    if feature_no_int != None and int(feature_no_int) > 0 and int(feature_no_int) <= feature_count:
        max_features = int(feature_no_int)
    else:
        max_features = DEFAULT_DISPLAY_COUNT
    
    local_shap_values = shap_explainer(current_input_vals)
    local_shap_values = shap.Explanation(values=local_shap_values.values,
                                         base_values=local_shap_values.base_values, 
                                         data=local_shap_values.data,
                                         feature_names=feature_list)
    
    # Build the matplotlib figure
    fig = plt.figure(figsize=(10, 5))
    match plottype:
        case 'local_bar':
            shap.plots.bar(local_shap_values[0], max_display=max_features, show=False)
        case 'waterfall':
            shap.plots.waterfall(local_shap_values[0], max_display=max_features, show=False)
    
    # Save the figure to a temporary buffer
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', format="png")

    # Embed the result in the html output
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'

    return html.Img(src=fig_bar_matplotlib, style={'width':'90%'})


# --- nWhy SHAP Plots -----------------------------------------------------------------------------
@app.callback(
    Output(component_id='nWhy_shap_out', component_property='children'),
    Output(component_id='nWhy_shap_figure', component_property='src'), 
    [   Input(component_id='predict_button', component_property='n_clicks'),
        Input(component_id='DEFAULT_nWHY_SHAP_CUTOFF_input', component_property='value'),
        Input(component_id='nWhy_shap_plottype_dropdown', component_property='value'),
        Input(component_id='nWhy_shap_featureno_input', component_property='value')
    ])
def update_nWhy_shapplot(n_clicks, cutoff, plottype, feature_no_int):
    global DEFAULT_nWHY_SHAP_CUTOFF
    if cutoff != None and float(cutoff) >= 0.0 and float(cutoff) <= 1.0:
        DEFAULT_nWHY_SHAP_CUTOFF = float(cutoff) 
    else:
        DEFAULT_nWHY_SHAP_CUTOFF = DEFAULT_nWHY_SHAP_CUTOFF

    if feature_no_int != None and int(feature_no_int) > 0 and int(feature_no_int) <= feature_count:
        max_features = int(feature_no_int)
    else:
        max_features = DEFAULT_DISPLAY_COUNT

    # Get the SHAP values
    shap_values_to_use = global_shap_values
    if plottype == 'local_bar':
        local_shap_values = shap_explainer(current_input_vals)
        local_shap_values = shap.Explanation(values=local_shap_values.values,
                                             base_values=local_shap_values.base_values, 
                                             data=local_shap_values.data,
                                             feature_names=feature_list)
        shap_values_to_use = local_shap_values

    # Order the features after their min shap value
    # TODO: Fix the order for the global plot (which seems not to be sorted correctly)
    agsv = np.abs(shap_values_to_use.values[0])
    df_sort = pd.DataFrame({'features': feature_list, 'shap vals': agsv})
    df_sort = df_sort.sort_values('shap vals')
    features_list = df_sort['features'].values
    col2num = {col: i for i, col in enumerate(feature_list)}
    order = list(map(col2num.get, features_list))

    # Get least influental features as list elements
    list_elements = []
    for i in range(len(features_list)):
        if  df_sort['shap vals'].values[i] <= DEFAULT_nWHY_SHAP_CUTOFF:
            li = html.Li(features_list[i])
            list_elements.append(li)
    if len(list_elements) == 0:
        list_elements.append(html.Li(" - "))
    
    # Build the matplotlib figure
    fig = plt.figure(figsize=(10, 5))
    match plottype:
        case 'global_bar':
            shap.plots.bar(shap_values_to_use, max_display=max_features, order=order, show=False)
        case 'local_bar':
            shap.plots.bar(shap_values_to_use[0], max_display=max_features, order=order, show=False)
    
    # Save the figure to a temporary buffer
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', format="png")

    # Embed the result in the html output
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'

    return list_elements, fig_bar_matplotlib


# --- Why Local LIME Plots ------------------------------------------------------------------------
@app.callback(
    Output(component_id='why_local_lime_figure', component_property='children'),
    [   Input('predict_button', 'n_clicks'),
        Input(component_id='why_local_lime_featureno_input', component_property='value')
    ],
    prevent_initial_call=True)
def update_why_local_limeplot(n_clicks, feature_no_int):
    if feature_no_int != None and int(feature_no_int) > 0 and int(feature_no_int) <= feature_count:
        max_features = int(feature_no_int)
    else:
        max_features = DEFAULT_DISPLAY_COUNT
    
    # Get the LIME explanation (it needs a 1D array of values)
    exp = lime_explainer.explain_instance(np.array(current_input_vals[0]), model.predict,
                                          num_features=max_features, top_labels=3)

    # Guess the needed height of the plot
    if max_features <= 2:
        height_val = 200
    else:
        height_val = 80 + (max_features * 50)
    height_str = str(height_val) + "px"
    
    # Get the Lime plot
    # NOTE: Default is as_html(labels=None, predict_proba=True, show_predicted_value=True, **kwargs)
    plot = html.Iframe(srcDoc=exp.as_html(), width='100%', height=height_str,
                       style={'border': '1px #d3d3d3 solid'})

    return plot


# --- Anchor Table --------------------------------------------------------------------------------
@app.callback(
    Output(component_id='anchor_why_out', component_property='children'), 
    [ Input('predict_button', 'n_clicks'),
      Input('anchor_why_format_method_input', 'value'),
      Input('anchor_threshold_input', 'value')
    ])
def update_de_anchor_table(n_clicks, format_type, anchor_threshold):
    if anchor_threshold != None and float(anchor_threshold) > 0.0 and float(anchor_threshold) <= 1.0:
        threshold = float(anchor_threshold)
    else:
        threshold = DEFAULT_ANCHOR_THRESHOLD
    
    # TODO: Not sure if this works with MLPReg model
    anchor_explanation = anchor_explainer.explain_instance(np.array(current_input_vals),
                                                    model.predict, threshold=threshold)
    anchor_rules = anchor_explanation.names()
    print("----- Anchors ------")
    print("Threshold =", threshold, ", Rules =", anchor_rules)
    
    if format_type == 0:
        rule_table_list = IXAII_functions.get_anchor_html_table_list(anchor_rules)
        return dbc.Table( [ html.Tbody(rule_table_list, id='anchor_rule_tablebody_display') ],
                          bordered=False)
    elif format_type == 1:
        anchor_box_plot = IXAII_functions.build_anchor_box_plot(feature_ranges, anchor_rules)
        return dcc.Graph(figure=anchor_box_plot, id='anchor_why_graph')
    else:
        return []


# --- DiCE Why Not Table --------------------------------------------------------------------------
@app.callback(
    Output(component_id='dice_why_not_out', component_property='children'), 
    [ Input('dice_why_not_generation_button', 'n_clicks'),
      State('dice_why_not_no_input', 'value'),
      State('dice_why_not_threshold_input', 'value'),
      State('dice_why_not_features_to_vary_input', 'value'),
      State('dice_why_not_format_method_input', 'value')
    ],
    prevent_initial_call=True)
def update_dice_why_not(n_clicks, example_no, threshold, varying_features_input, format_type):
    if example_no != None and int(example_no) > 0 and int(example_no) <= DEFAULT_DICE_MAX_EXAMPLE_COUNT:
        example_no = int(example_no)
    else:
        example_no = DEFAULT_DICE_EXAMPLE_COUNT

    if threshold != None and float(threshold) >= 0.0 and float(threshold) <= 1.0:
        threshold = float(threshold) 
    else:
        threshold = DEFAULT_DICE_TRESHOLD

    # Build varying_features_list and complete_feature_list for future use
    varying_features_list, complete_feature_list = IXAII_functions.get_varying_and_complete_feature_list(feature_list, 
                                                                                    varying_features_input)

    # Get the target range of the examples (and the example type for the table header)
    target_range = []
    current_pred = round(y_pred[0])
    split_target_range = False
    
    # Other range than the current pred
    if current_pred == 0: target_range = [0.5, 2.45]
    elif current_pred == 1: split_target_range = True
    elif current_pred == 2: target_range = [0.0, 1.45]

    # Init feature value matrix with schema:
    # [ ['feature1 name', cf1_val, cf2_val, ...], ['feature2 name', cf1_val, cf2_val, ...], ...]
    feature_value_matrix = []

    # Get the DiCE Explanation
    input_values_df = pd.DataFrame(current_input_vals, columns=feature_list)
    print("WHY NOT - DiCE Input Values:")
    print(input_values_df)
    if split_target_range:
        # Get explanations for both ranges
        cf_no_1 = math.ceil(example_no/2)
        print("Creating", cf_no_1, "CFs in Range 0.0 - 0.45")
        dice_exp1 = dice_explainer.generate_counterfactuals(
            query_instances = input_values_df,
            total_CFs = cf_no_1,
            features_to_vary  = varying_features_list,
            stopping_threshold = threshold,
            desired_range = [0.0, 0.45]
        )
        cf_no_2 = math.floor(example_no/2)
        print("Creating", cf_no_2, "CFs in Range 1.5 - 2.45")
        dice_exp2 = dice_explainer.generate_counterfactuals(
            query_instances = input_values_df,
            total_CFs = cf_no_2,
            features_to_vary  = varying_features_list,
            stopping_threshold = threshold,
            desired_range = [1.5, 2.45]
        )
        # Join the example value lists
        cfs_as_df_1 = dice_exp1.cf_examples_list[0].final_cfs_df
        cfs_as_df_2 = dice_exp2.cf_examples_list[0].final_cfs_df
        for feature in complete_feature_list:
            feature_of_cfs_value_list = [feature]
            for i in range (cf_no_1):
                feature_of_cfs_value_list.append(cfs_as_df_1[feature].values[i])
            for i in range (cf_no_2):
                feature_of_cfs_value_list.append(cfs_as_df_2[feature].values[i])
            feature_value_matrix.append(feature_of_cfs_value_list)
    else:
        print("Creating", example_no, "CFs in Range", target_range[0], "-", target_range[1])
        dice_when_exp = dice_explainer.generate_counterfactuals(
            query_instances = input_values_df,
            total_CFs = example_no,
            features_to_vary  = varying_features_list,
            stopping_threshold = threshold,
            desired_range = target_range
        )
        cfs_as_df = dice_when_exp.cf_examples_list[0].final_cfs_df
        for feature in complete_feature_list:
            feature_of_cfs_value_list = [feature]
            for i in range (example_no):
                feature_of_cfs_value_list.append(cfs_as_df[feature].values[i])
            feature_value_matrix.append(feature_of_cfs_value_list)

    if format_type == 0:
        dice_table = IXAII_functions.build_dice_table(example_no, complete_feature_list,
                    feature_value_matrix, pre_target_str, target_list, varying_features_list)
        return dbc.Table(dice_table, id='dice_why_not_table_display', bordered=False,
                         className='mt-4 table-hover')
    elif format_type == 1:
        dice_plot = IXAII_functions.build_dice_plot(example_no, feature_list, current_input_vals,
                                          feature_value_matrix)
        return dcc.Graph(figure=dice_plot, id='dice_why_not_plot_display')
    else:
        return []


# --- DiCE What If Table --------------------------------------------------------------------------
@app.callback(
    Output(component_id='dice_what_if_out', component_property='children'), 
    [ Input('dice_what_if_generation_button', 'n_clicks'),
      State('dice_what_if_no_input', 'value'),
      State('dice_what_if_threshold_input', 'value'),
      State('dice_what_if_features_to_vary_input', 'value'),
      State('dice_what_if_format_method_input', 'value')
    ],
    prevent_initial_call=True)
def update_dice_what_if(n_clicks, example_no, threshold, varying_features_input, format_type):
    if example_no != None and int(example_no) > 0 and int(example_no) <= DEFAULT_DICE_MAX_EXAMPLE_COUNT:
        example_no = int(example_no)
    else:
        example_no = DEFAULT_DICE_EXAMPLE_COUNT

    if threshold != None and float(threshold) >= 0.0 and float(threshold) <= 1.0:
        threshold = float(threshold)
    else:
        threshold = DEFAULT_DICE_TRESHOLD

    # Build varying_features_list and complete_feature_list for future use
    varying_features_list, complete_feature_list = IXAII_functions.get_varying_and_complete_feature_list(feature_list, 
                                                                                    varying_features_input)

    # What If = Factual Examples = Complete range
    target_range = [0.0, 2.5]

    # Init feature value matrix with schema:
    # [ ['feature1 name', cf1_val, cf2_val, ...], ['feature2 name', cf1_val, cf2_val, ...], ...]
    feature_value_matrix = []

    # Get the DiCE Explanation
    input_values_df = pd.DataFrame(current_input_vals, columns=feature_list)
    print("DiCE Input Values:")
    print(input_values_df)
    dice_what_if_exp = dice_explainer.generate_counterfactuals(
        query_instances = input_values_df,
        total_CFs = example_no,
        features_to_vary  = varying_features_list,
        stopping_threshold = threshold,
        desired_range = target_range
    )

    # Copy the example values to the matirx
    cfs_as_df = dice_what_if_exp.cf_examples_list[0].final_cfs_df
    for feature in complete_feature_list:
        feature_of_cfs_value_list = [feature]
        for i in range (example_no):
            feature_of_cfs_value_list.append(cfs_as_df[feature].values[i])
        feature_value_matrix.append(feature_of_cfs_value_list)

    if format_type == 0:
        dice_table = IXAII_functions.build_dice_table(example_no, complete_feature_list,
                    feature_value_matrix, pre_target_str, target_list, varying_features_list)
        return dbc.Table(dice_table, id='dice_what_if_table_display', bordered=False,
                         className='mt-4 table-hover')
    elif format_type == 1:
        dice_plot = IXAII_functions.build_dice_plot(example_no, feature_list, current_input_vals,
                                          feature_value_matrix)
        return dcc.Graph(figure=dice_plot, id='dice_what_if_plot_display')
    else:
        return []


# --- DiCE When Table -----------------------------------------------------------------------------
@app.callback(
    Output(component_id='dice_when_out', component_property='children'), 
    [ Input('dice_when_generation_button', 'n_clicks'),
      State('dice_when_target_class_input', 'value'),
      State('dice_when_no_input', 'value'),
      State('dice_when_threshold_input', 'value'),
      State('dice_when_features_to_vary_input', 'value'),
      State('dice_when_format_method_input', 'value')
    ],
    prevent_initial_call=True)
def update_dice_when(n_clicks, target_class, example_no, threshold, varying_features_input, format_type):
    if target_class != None and int(target_class) >= 0 and int(target_class) <= len(target_list)-1:
        target_class = int(target_class) 
    else:
        target_class = DEAFUALT_WHEN_TARGET_CLASS

    if example_no != None and int(example_no) > 0 and int(example_no) <= DEFAULT_DICE_MAX_EXAMPLE_COUNT:
        example_no = int(example_no)
    else:
        example_no = DEFAULT_DICE_EXAMPLE_COUNT

    if threshold != None and float(threshold) >= 0.0 and float(threshold) <= 1.0:
        threshold = float(threshold)
    else:
        threshold = DEFAULT_DICE_TRESHOLD

    target_range = []

    # Check if a target class was given 
    if target_class == 0:
        target_range = [0.0, 0.45]
    elif target_class == 1:
        target_range = [0.5, 1.45]
    elif target_class == 2:
        target_range = [1.5, 2.45]
    else:
        return []

    # Build varying_features_list and complete_feature_list for future use
    varying_features_list, complete_feature_list = IXAII_functions.get_varying_and_complete_feature_list(feature_list, 
                                                                                    varying_features_input)

    # Init feature value matrix with schema:
    # [ ['feature1 name', cf1_val, cf2_val, ...], ['feature2 name', cf1_val, cf2_val, ...], ...]
    feature_value_matrix = []

    # Get the DiCE Explanation
    input_values_df = pd.DataFrame(current_input_vals, columns=feature_list)
    dice_when_exp = dice_explainer.generate_counterfactuals(
        query_instances = input_values_df,
        total_CFs = example_no,
        features_to_vary  = varying_features_list,
        stopping_threshold = threshold,
        desired_range = target_range
    )

    # Copy the example values to the matirx
    cfs_as_df = dice_when_exp.cf_examples_list[0].final_cfs_df
    for feature in complete_feature_list:
        feature_of_cfs_value_list = [feature]
        for i in range (example_no):
            feature_of_cfs_value_list.append(cfs_as_df[feature].values[i])
        feature_value_matrix.append(feature_of_cfs_value_list)

    if format_type == 0:
        dice_table = IXAII_functions.build_dice_table(example_no, complete_feature_list,
                    feature_value_matrix, pre_target_str, target_list, varying_features_list)
        return dbc.Table(dice_table, id='dice_when_table_display', bordered=False,
                         className='mt-4 table-hover')
    elif format_type == 1:
        dice_plot = IXAII_functions.build_dice_plot(example_no, feature_list, current_input_vals,
                                          feature_value_matrix)
        return dcc.Graph(figure=dice_plot, id='dice_when_plot_display')
    else:
        return []





# --- RUN ----------------------------------------------------------------------------------------------
app.run_server(debug=True)