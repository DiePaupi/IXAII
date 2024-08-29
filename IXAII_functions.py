# Here are helping function for the IXAII defined

from dash import html

import pandas as pd
import numpy as np
import math
import re
import plotly.graph_objects as go
import plotly.express as px

from sklearn.neural_network import MLPRegressor
import xgboost as xgb

import shap
import lime
import lime.lime_tabular
from anchor import anchor_tabular
import dice_ml



########################################################################################################
# 1) Import dataset 
########################################################################################################

def get_data_df(dataset, feature_list):
    # 1.1) Create a pandas DataFrame from the dataset ----------------------------------------
    dataset_df = pd.DataFrame(dataset.data, columns=feature_list)
    dataset_df['target'] = dataset.target
    X_df = dataset_df.drop('target', axis = 1)

    # 1.2) Target_names matchen --------------------------------------------------------------
    target_mapping = {x: dataset.target_names[x] for x in range(len(dataset.target_names))}
    dataset_df['target_names'] = dataset_df['target'].map(target_mapping)

    return dataset_df, X_df


def get_outcomes(target_names):
    target_list = []
    for i in range(len(target_names)):
        temp_str = str(i) + ": " + str(target_names[i])
        target_list.append(temp_str)
    
    table_header = [html.Thead(html.Tr([
        html.Th("ID"), html.Th("Species")
    ], className='table-primary'))]
    
    table_body = []
    for i in range(len(target_names)):
        row = html.Tr([
            html.Td(str(i)), html.Td(target_names[i])
        ])
        table_body.append(row)
        
    return target_list, table_header + [html.Tbody(table_body)]



########################################################################################################
# 2) Get prototypical input values
########################################################################################################

def get_avg_input_set(dataset_df, data_row_count):
    avg_data_values = []
    for feature in dataset_df.columns:
        if feature != 'target' and feature != 'target_names':
            data_column = dataset_df[feature]
            avg_feature_value = round(sum(data_column) / data_row_count, 3)

            #print("Avg. feature value of feature", feature, "=", avg_feature_value)
            avg_data_values.append(avg_feature_value)
    
    return [avg_data_values]


def get_prototypical_input_sets(dataset_df, target_names, feature_list):
    outcome_prototype_values = []
    # NOTE: This loop is based on the assumption that the outcomes are integers starting at 0
    for i_outcome in range(len(target_names)):
        # Filter rows where the 'target' column equals 'i_outcome'
        filtered_df = dataset_df[dataset_df['target'] == i_outcome]
        # Drop the 'target' column
        filtered_df = filtered_df.drop(columns=['target_names']).drop(columns=['target'])
        # Calculate the 50% median for all other features
        median_values = filtered_df.median(numeric_only=True)
        # Add this protoypical value list for outcomi i to the overall list
        outcome_prototype_values.append(median_values)

    # Build table header
    prototype_table_header = []
    prototype_table_header.append(html.Th("Features"))
    for i in range(len(target_names)):
        prototype_table_header.append(html.Th(target_names[i]))
    prototype_table_header = [html.Thead([html.Tr(prototype_table_header, className='table-primary')])]

    # Build table body
    prototype_table_body = []
    for feature_i in range(len(feature_list)):
        table_row = [html.Td(feature_list[feature_i])]
        
        for outome_i in range(len(target_names)):
            table_row.append(html.Td(outcome_prototype_values[outome_i][feature_i]))

        prototype_table_body.append(html.Tr(table_row))
    
    return prototype_table_header + [html.Tbody(prototype_table_body)]



########################################################################################################
# 3) Get model and explainer according to the chosen ML method
########################################################################################################

def get_model(ml_method):
    match ml_method:
        case 'MLPReg':
            model = MLPRegressor(hidden_layer_sizes=(6,), random_state=0, max_iter=10000)
        case 'XGBReg':
            model = xgb.XGBRegressor(objective="reg:squarederror")
    return model

def get_shap_explainer(ml_method, model, X_train):
    match ml_method:
        case 'MLPReg':
            shap_explainer = shap.KernelExplainer(model=model.predict, data=X_train, link="identity")
        case 'XGBReg':
            shap_explainer = shap.Explainer(model)
    return shap_explainer


# NOTE: Since the model is not needed for LIME, it doesn't matter
def get_lime_explainer(X_train, feature_list, target_name_list):
    # NOTE: See https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_tabular
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_list,
                        class_names=target_name_list, verbose=True, mode='regression',
                        discretize_continuous=False)
    return lime_explainer


# NOTE: Since the model is not needed for Anchor, it doesn't matter
def get_achnor_explainer(X_np, feature_list, target_name_list):
    # NOTE: When using this explainer to make a prediction, the input needs to be an np-array
    anchor_explainer = anchor_tabular.AnchorTabularExplainer(
        target_name_list, feature_list, X_np)
    return anchor_explainer


def get_dice_explainer(ml_method, model, dataset_df, feature_list):
    dice_d = dice_ml.Data(dataframe = dataset_df.drop('target_names', axis = 1),
                            continuous_features = feature_list,
                            outcome_name = 'target')
    
    match ml_method:
        case 'MLPReg':
            # TODO: Not sure if this works with MLPReg model
            dice_m = dice_ml.Model(model = model.predict, backend='sklearn', model_type='regressor')
        case 'XGBReg':
            dice_m = dice_ml.Model(model = model, backend='sklearn', model_type='regressor')

    return dice_ml.Dice(dice_d, dice_m, method='genetic')



########################################################################################################
# 4) Get explainer specific things
########################################################################################################

def get_global_shap_values(shap_explainer, X_df, X_train, feature_list):
    global_shap_values = shap_explainer(X_train)
    global_shap_values = shap.Explanation(values=global_shap_values.values,
                        base_values=global_shap_values.base_values, data=global_shap_values.data,
                        feature_names=feature_list)
    
    # NOTE: Solution taken from "TTFrance" in https://github.com/shap/shap/issues/632
    df_shap_values = pd.DataFrame(data=global_shap_values.values, columns=X_df.columns)
    global_shap_importance_df = pd.DataFrame(columns=['feature','importance'])

    for col in df_shap_values.columns:
        importance = df_shap_values[col].abs().mean()
        global_shap_importance_df.loc[len(global_shap_importance_df)] = [col,importance]

    global_shap_importance_df = global_shap_importance_df.sort_values('feature', ascending=True)

    return global_shap_values, global_shap_importance_df


def return_lime_exp_weights(explainer):
    exp_map = explainer.as_map()[1]
    exp_map = sorted(exp_map, key=lambda x: x[0])
    exp_weight = [x[1] for x in exp_map]
    return exp_weight


def get_lime_values(lime_explainer, model, X_train, feature_list):
    # TODO: Not sure if this works with MLPReg model
    all_weights = []
    for x in X_train:
        # NOTE: For classification use: model.predict_proba
        loc_exp = lime_explainer.explain_instance(np.array(x), model.predict,
                    num_features=len(feature_list), labels=feature_list)
        exp_weight = return_lime_exp_weights(loc_exp)
        all_weights.append(exp_weight)
    lime_weights = pd.DataFrame(data=all_weights, columns=feature_list)

    # Get the absolut mean and sort
    lime_importance_df = lime_weights.abs().mean(axis=0)
    lime_importance_df = pd.DataFrame(data={'feature': lime_importance_df.index,
                                            'abs_mean': lime_importance_df})
    lime_importance_df = lime_importance_df.sort_values('feature', ascending=True)

    return lime_importance_df


def get_global_importance_plot(global_shap_importance_df, lime_importance_df):
    global_importance_figure = go.Figure()
    global_importance_figure.add_trace(go.Bar(
        x = global_shap_importance_df['feature'],
        y = global_shap_importance_df['importance'],
        name = "SHAP",
        marker_color = 'rgb(55, 83, 109)'
    ))
    global_importance_figure.add_trace(go.Bar(
        x = lime_importance_df['feature'],
        y = lime_importance_df['abs_mean'],
        name = "LIME",
        marker_color = 'rgb(26, 118, 255)'
    ))
    return global_importance_figure


def get_anchor_html_table_list(anchor_rules):
    rule_table_list = []
    for rule in anchor_rules:
        elements = re.split(r"\s", rule)
        # 5 Entry Structure: ['sepal', 'length', '(cm)', '<=', '6.40']
        if len(elements) == 5:
            table_row = html.Tr([
                html.Td("  "),
                html.Td(elements[0] + " " + elements[1] + " " + elements[2]),
                html.Td(elements[3] + " " + elements[4])
            ])
            rule_table_list.append(table_row)
        # 7 Entry Structure: ['0.30', '<', 'petal', 'width', '(cm)', '<=', '1.30']
        elif len(elements) == 7:
            table_row = html.Tr([
                html.Td(elements[0] + " " + elements[1]),
                html.Td(elements[2] + " " + elements[3] + " " + elements[4]),
                html.Td(elements[5] + " " + elements[6])
            ])
            rule_table_list.append(table_row)
        else:
            rule_table_list.append(html.Tr([html.Td("Error: Rule structure not recognized")]))

    return rule_table_list


def build_anchor_box_plot(feature_ranges, anchor_rules):
    # Create the box plot
    fig = go.Figure()

    for feature_rule in anchor_rules:
        elements = re.split(r"\s", feature_rule)
        # INFO: 5 Entry Structure: ['sepal', 'length', '(cm)', '<=', '6.40']
        if len(elements) == 5:
            feature_name = elements[0] + " " + elements[1]
            if "<=" == elements[3] or "<" == elements[3]:
                rule_min_value = "-"
                rule_max_value = float(elements[4])
            else: 
                rule_min_value = float(elements[4])
                rule_max_value = "-"
        # INFO: 7 Entry Structure: ['0.30', '<', 'petal', 'width', '(cm)', '<=', '1.30']
        elif len(elements) == 7:
            rule_min_value = float(elements[0])
            #rule_min_symbol = elements[1]
            feature_name = elements[2] + " " + elements[3]
            #rule_max_symbol = elements[5]
            rule_max_value = float(elements[6])
        else:
            feature_name = "unkown"
            rule_min_value = 0
            rule_max_value = 0

        # Check the feature name to get the range
        # INFO: Iris feature ranges are = [ [3.0, 9.0], [1.0, 6.0], [0.0, 8.0], [0.0, 4.0]]
        match feature_name:
            case 'sepal length (cm)':
                range_min = feature_ranges[0][0]
                range_max = feature_ranges[0][0]
            case 'sepal width (cm)':
                range_min = feature_ranges[1][0]
                range_max = feature_ranges[1][0]
            case 'petal length (cm)':
                range_min = feature_ranges[2][0]
                range_max = feature_ranges[2][0]
            case 'petal width (cm)':
                range_min = feature_ranges[3][0]
                range_max = feature_ranges[3][0]
            case _:
                range_min = 0
                range_max = 0
        
        # Overwrite the "-" values with the ranges for 5-element rules
        if str(rule_min_value) == "-":
            rule_min_value = range_min
        if str(rule_max_value) == "-":
            rule_max_value = range_max

        # Add the trace of the current rule to the box plot
        fig.add_trace(go.Box(
            y=[rule_min_value, rule_max_value],
            boxpoints=False,            # Hide individual data points
            name=feature_name,          # Name of the box
            marker=dict(color='rgba(70, 175, 244, 0.6)'),   # Box color
            line=dict(color='rgba(70, 175, 244, 1)'),       # Line color
            width=0.5                                   # Width of the box
        ))

        # Customize layout and remove the legend
        fig.update_layout(
            yaxis_title = "cm",
            showlegend = False
        )
    
    return fig


def get_dice_varying_features_checkboxes(feature_list):
    dice_feature_variation_checkboxes = []
    for i in range(len(feature_list)):
        dice_feature_variation_checkboxes.append({'label': feature_list[i], 'value': i})
    return dice_feature_variation_checkboxes



########################################################################################################
# X) Helping Functions
########################################################################################################

def get_varying_and_complete_feature_list(feature_list, varying_features_input):
    varying_features_list = []
    complete_feature_list = []  # This needs to be a deep copy!
    for i in range(len(feature_list)):
        complete_feature_list.append(feature_list[i])
        if i in varying_features_input:
            varying_features_list.append(feature_list[i])
    complete_feature_list.append('target')

    return varying_features_list, complete_feature_list


def build_dice_table(example_no, complete_feature_list, feature_value_matrix,
                     pre_target_str, target_list, varying_features_list):

    # Build table header
    dice_cf_table_header = []
    dice_cf_table_header.append(html.Th("Features"))
    for i in range(example_no):
        header_str = "Example " + str(i+1)
        dice_cf_table_header.append(html.Th(header_str))
    dice_cf_table_header = [html.Tr(dice_cf_table_header, className='table-primary')] 

    # Build table body
    dice_cf_table_body = []
    for f_no in range(len(complete_feature_list)):
        table_row = []
        feature_of_cfs_value_list = feature_value_matrix[f_no]
        feature_name = feature_of_cfs_value_list[0]

        if feature_name == 'target':
            table_row.append(html.Th(pre_target_str))
            for i in range(1, example_no+1):
                outcome_as_int = round(float(feature_of_cfs_value_list[i]))
                table_row.append(html.Td(target_list[outcome_as_int]))
            dice_cf_table_body.append(html.Tr(table_row, className='table-primary'))
        else:
            table_row.append(html.Th(feature_name))
            for i in range(1, example_no+1):
                table_row.append(html.Td(str(feature_of_cfs_value_list[i])))

            if feature_name in varying_features_list:
                dice_cf_table_body.append(html.Tr(table_row, className='table-info'))
            else:
                dice_cf_table_body.append(html.Tr(table_row))
    
    return [html.Thead(dice_cf_table_header), html.Tbody(dice_cf_table_body)]


def build_dice_plot(example_no, feature_list, current_input_vals, feature_value_matrix):
    # INFO: feature_value_matrix schema:
    # -> [ ['feature1 name', cf1_val, cf2_val, ...], ['feature2 name', cf1_val, cf2_val, ...], ...]

    dice_figure = go.Figure()

    # Add original inputs -------------------------------------------------------------------------
    original_inputs_df = pd.DataFrame(columns=['feature','value'])

    for feature_i in range(len(feature_list)):
        value = current_input_vals[0][feature_i]
        original_inputs_df.loc[len(original_inputs_df)] = [feature_list[feature_i], value]

    original_inputs_df = original_inputs_df.sort_values('feature', ascending=True)

    dice_figure.add_trace(go.Bar(
        x = original_inputs_df['feature'],
        y = original_inputs_df['value'],
        name = "Your Input Values",
        marker_color = 'rgb(55, 83, 109)'
    ))

    # Add example inputs --------------------------------------------------------------------------
    for example_i in range(example_no):

        example_inputs_df = pd.DataFrame(columns=['feature','value'])

        # Get the feature values
        for feature_i in range(len(feature_list)):
            value = feature_value_matrix[feature_i][example_i+1]
            example_inputs_df.loc[len(example_inputs_df)] = [feature_list[feature_i], value]
        example_inputs_df = example_inputs_df.sort_values('feature', ascending=True)
        
        # Get the outcome and its color
        outcome_i = len(feature_value_matrix)-1
        outcome_value = round(feature_value_matrix[outcome_i][example_i+1])
        match outcome_value:
            case 0:
                color = 'rgb(70, 175, 244)' # Light Blue
            case 1:
                color = 'rgb(20, 175, 170)' # Turquoise
            case 2:
                color = 'rgb(30, 70, 200)'  # Medium Blue
            case _:
                color = 'rgb(240, 140, 30)' # Orange



        dice_figure.add_trace(go.Bar(
            x = example_inputs_df['feature'],
            y = example_inputs_df['value'],
            name = "Example " + str(example_i) + ", Species " + str(outcome_value),
            marker_color = color
        ))
    
    dice_figure.update_layout(barmode='group', legend_title_text = "Inputs", xaxis_tickangle=-45)
    return dice_figure
