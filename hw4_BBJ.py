import pandas as pd
import numpy as np
import os
import refinitiv.dataplatform.eikon as ek
import refinitiv.data as rd
from dash import Dash, html, dcc, dash_table, Input, Output, State
from datetime import datetime, date, timedelta
import dash_bootstrap_components as dbc
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

from math import log
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

#####################################################

ek.set_app_key(os.getenv('EIKON_API'))

# ----html starts----

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

percentage = dash_table.FormatTemplate.percentage(3)

controls = dbc.Card(
    [
        dcc.Input(id='asset-id', type='text', value="IVV",
                  style={'display': 'inline-block',
                         'border': '1px solid black'}),
        dbc.Row([
            dcc.DatePickerRange(
                id='my-date-picker-range',
                min_date_allowed=date(2020, 1, 3),
                max_date_allowed=date(2023, 3, 23),
                # initial_visible_month=date(2023, 1, 30),
                start_date=date(2020, 1, 3),
                end_date=date(2023, 3, 23)
            )
        ]),
        dbc.Row(html.Button('QUERY Refinitiv', id='run-query', n_clicks=0)),
        dbc.Row([
            html.H5('Asset:',
                    style={'display': 'inline-block', 'margin-right': 20}),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α1"), html.Th("n1")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha1-id',
                                    type='number',
                                    value=-0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='day1-id',
                                    type='number',
                                    value=3,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            ),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α2"), html.Th("n2")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha2-id',
                                    type='number',
                                    value=0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='day2-id',
                                    type='number',
                                    value=5,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            )
        ]),
        dbc.Row([
            dcc.Markdown('''Press to display the order info:'''),
            html.Button('Submit', id='update_parameter', n_clicks=0)
        ]),
        html.Br(),
        dbc.Row([
            html.Label('''Amount for training set:'''),
            dbc.Input(id='n-train',
                      type='number',
                      value=30,
                      max=700,
                      min=5,
                      step=1
                      ),
            html.Button('Predict', id='run-predict', n_clicks=0)
        ])
    ],
    body=True
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(
                    # Put your reactive graph here as an image!
                    html.Img(src=app.get_asset_url('reactive_graph.png'), alt="reactive graph", width=720, height=450),
                    md=8
                )
            ],
            align="center",
        ),
        html.Br(),
        dbc.Row([
            html.H2('Raw Data'),
            dash_table.DataTable(
                id="ivv_prc_data",
                page_action='none',
                style_table={'height': '300px', 'overflowY': 'auto'}
                # ), style={'display': 'none'}
            ),
        ]),
        html.Br(),
        dbc.Row([
            html.H2('Blotter'),
            dash_table.DataTable(
                id="all_orders-tbl",
                page_action='none',
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
        ]),
        html.Br(),
        dbc.Row([
            html.H2('Ledger'),
            dash_table.DataTable(
                id="ledger-tbl",
                page_action='none',
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
        ]),
        html.Br(),
        dbc.Row([
            html.H2('Predict Ledger'),
            dash_table.DataTable(
                id="predict-tbl",
                page_action='none',
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
        ]),
        html.Br(),
        dbc.Row([
            html.P(id='predict-text', children="")
        ]),

    ],
    fluid=True
)


# ----html ends----
# next_business_day = datetime.now().strftime("%Y-%m-%d")
# ivv_prc_data = pd.DataFrame()

# Parameters:
# alpha1 = -0.01
# day1 = 3
# alpha2 = 0.01
# day2 = 5
# asset = "IVV"

@app.callback(
    Output("ivv_prc_data", 'data'),
    Input("run-query", "n_clicks"),
    [State('asset-id', 'value'),
     State('my-date-picker-range', 'start_date'),
     State('my-date-picker-range', 'end_date')],
    prevent_initial_call=True
)
def update_parameters(n_clicks, asset, start_date, end_date):
    ivv_prc, ivv_prc_err = ek.get_data(
        instruments=[asset],
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    ivv_prc['Date'] = pd.to_datetime(ivv_prc['Date']).dt.date
    ivv_prc.drop(columns='Instrument', inplace=True)
    # print(ivv_prc.pivot_table(index='Date').to_dict('records'))
    # ivv_prc = ivv_prc.pivot_table(
    #         index='Date', columns='Instrument'
    #     ).to_dict('records')

    return ivv_prc.to_dict('records')


@app.callback(  # when history table changes, return table will change
    Output("all_orders-tbl", "data"),
    Input("update_parameter", "n_clicks"),
    [State("ivv_prc_data", 'data'), State('asset-id', 'value'), State('alpha1-id', 'value'),
     State('day1-id', 'value'), State('alpha2-id', 'value'), State('day2-id', 'value')],
    prevent_initial_call=True
)
def output_orders(n_clicks, ivv_prc, asset, alpha1, day1, alpha2, day2):
    # submitted entry orders
    ivv_prc_df = pd.DataFrame(ivv_prc)
    ivv_prc_df['Date'] = pd.to_datetime(ivv_prc_df['Date']).dt.date

    rd.open_session()
    next_business_day = rd.dates_and_calendars.add_periods(
        start_date=ivv_prc_df['Date'].iloc[-1].strftime("%Y-%m-%d"),
        period="1D",
        calendars=["USA"],
        date_moving_convention="NextBusinessDay",
    )
    rd.close_session()

    submitted_entry_orders = pd.DataFrame({
        "trade_id": range(1, ivv_prc_df.shape[0]),
        "date": list(pd.to_datetime(ivv_prc_df["Date"].iloc[1:]).dt.date),
        "asset": asset,
        "trip": 'ENTRY',
        "action": "BUY",
        "type": "LMT",
        "price": round(
            ivv_prc_df['Close Price'].iloc[:-1] * (1 + alpha1),
            2
        ),
        'status': 'SUBMITTED'
    })

    # get the cancelled and filled entry orders
    temp_entry_orders = submitted_entry_orders.copy()
    for i in range(0, len(temp_entry_orders)):

        for j in range(1, day1 + 1):
            if i + j > len(ivv_prc_df) - 1:
                break
            if temp_entry_orders["price"].iloc[i] >= ivv_prc_df["Low Price"].iloc[i + j]:
                temp_entry_orders["status"].iloc[i] = "FILLED"
                temp_entry_orders["date"].iloc[i] = ivv_prc_df["Date"].iloc[i + j]
                break
        if temp_entry_orders["status"].iloc[i] != "FILLED":
            if i + j > len(ivv_prc) - 1:
                continue
            temp_entry_orders["status"].iloc[i] = "CANCELLED"
            temp_entry_orders["date"].iloc[i] = ivv_prc_df["Date"].iloc[i + day1]
    # live entry orders
    live_index = temp_entry_orders.index[temp_entry_orders["status"] == "SUBMITTED"]
    for i in live_index:
        temp_entry_orders["status"].iloc[i] = "LIVE"
        temp_entry_orders["date"].iloc[i] = pd.to_datetime(next_business_day).date()
    latest_live_entry_orders = pd.DataFrame({
        "trade_id": ivv_prc_df.shape[0],
        "date": pd.to_datetime(next_business_day).date(),
        "asset": asset,
        "trip": 'ENTRY',
        "action": "BUY",
        "type": "LMT",
        "price": round(ivv_prc_df['Close Price'].iloc[-1] * (1 + alpha1), 2),
        'status': 'LIVE'
    },
        index=[0]
    )

    entry_orders = pd.concat(
        [
            submitted_entry_orders,
            temp_entry_orders,
            latest_live_entry_orders
        ]
    ).sort_values(["trade_id", 'date'])
    entry_orders.reset_index(drop=True, inplace=True)

    # if the limit order filled, immediately submit exit orders
    submitted_exit_orders = entry_orders[entry_orders["status"] == "FILLED"].copy()
    submitted_exit_orders['trip'] = "EXIT"
    submitted_exit_orders['action'] = 'SELL'
    submitted_exit_orders['price'] = round(
        (1 + alpha2) * submitted_exit_orders['price'],
        2
    )
    submitted_exit_orders['status'] = "SUBMITTED"

    # get the cancelled and filled exit orders
    temp_exit_orders = submitted_exit_orders.copy()
    for i in range(0, len(temp_exit_orders)):
        # get the index in original data for the current date
        ori_index = ivv_prc_df.index[ivv_prc_df["Date"] == temp_exit_orders["date"].iloc[i]][0]

        for j in range(0, day2):
            if ori_index + j > len(ivv_prc_df) - 1:
                break
            if j == 0:
                if temp_exit_orders["price"].iloc[i] <= ivv_prc_df["Close Price"].iloc[ori_index + j]:
                    temp_exit_orders["status"].iloc[i] = "FILLED"
                    temp_exit_orders["date"].iloc[i] = ivv_prc_df["Date"].iloc[ori_index + j]
                    break
            else:
                if temp_exit_orders["price"].iloc[i] <= ivv_prc_df["High Price"].iloc[ori_index + j]:
                    temp_exit_orders["status"].iloc[i] = "FILLED"
                    temp_exit_orders["date"].iloc[i] = ivv_prc_df["Date"].iloc[ori_index + j]
                    break
        if temp_exit_orders["status"].iloc[i] != "FILLED":
            if ori_index + j > len(ivv_prc_df) - 1:
                continue
            temp_exit_orders["status"].iloc[i] = "CANCELLED"
            temp_exit_orders["date"].iloc[i] = ivv_prc_df["Date"].iloc[ori_index + day2 - 1]
    temp_exit_orders.reset_index(drop=True, inplace=True)
    live_index = temp_exit_orders.index[temp_exit_orders["status"] == "SUBMITTED"]
    for i in live_index:
        temp_exit_orders["status"].iloc[i] = "LIVE"
        temp_exit_orders["date"].iloc[i] = pd.to_datetime(next_business_day).date()

    # for the cancelled exit order, immediately issue a market order to sell
    exit_orders = temp_exit_orders.copy()  # define a global exit_orders
    if any(temp_exit_orders['status'] == 'CANCELLED'):
        submitted_exit_market_orders = temp_exit_orders[temp_exit_orders["status"] == "CANCELLED"].copy()
        submitted_exit_market_orders['type'] = "MKT"
        submitted_exit_market_orders['status'] = "SUBMITTED"
        submitted_exit_market_orders.reset_index(drop=True, inplace=True)
        for i in range(len(submitted_exit_market_orders)):
            submitted_exit_market_orders["price"].iloc[i] = ivv_prc_df[
                ivv_prc_df["Date"] == submitted_exit_market_orders['date'][i]
                ].copy()['Close Price']
        # These market order fill on the same day, at closing price
        filled_exit_market_orders = submitted_exit_market_orders.copy()
        filled_exit_market_orders['status'] = "FILLED"
        exit_market_orders = pd.concat(
            [
                submitted_exit_market_orders,
                filled_exit_market_orders
            ]
        ).sort_values(["trade_id", 'date'])
        exit_orders = pd.concat(
            [
                submitted_exit_orders,
                temp_exit_orders,
                exit_market_orders
            ]
        ).sort_values(["trade_id", 'date'])
    else:
        exit_orders = pd.concat(
            [
                submitted_exit_orders,
                temp_exit_orders
            ]
        ).sort_values(["trade_id", 'date'])
    exit_orders.reset_index(drop=True, inplace=True)

    # join entry_order, exit_order, exit_market_order together
    all_orders = pd.concat(
        [
            entry_orders,
            exit_orders
        ]
    ).sort_values(["trade_id", 'date'])
    all_orders.reset_index(drop=True, inplace=True)
    # all_orders.set_index('trade_id', inplace=True)
    all_orders.to_csv('blotter.csv')
    return all_orders.to_dict("records")


# predict function start

@app.callback(  # when history table changes, return table will change
    Output("ledger-tbl", "data"),
    Input("all_orders-tbl", "data"),
    prevent_initial_call=True
)
def ledger(blotter_dict):
    ledger_dict = {'trade_id': [], 'asset': [], 'dt_enter': [], 'dt_exit': [], 'success': [], 'n': [], 'rtn': []}
    enter_price = []
    pre_trade_id = 0
    blotter = pd.DataFrame(blotter_dict)
    # blotter['Date'] = pd.to_datetime(blotter['Date']).dt.date
    for index, row in blotter.iterrows():
        current_tid = row['trade_id']
        if current_tid != pre_trade_id:  # new trade
            ledger_dict['trade_id'].append(current_tid)
            ledger_dict['asset'].append(row['asset'])
            ledger_dict['dt_enter'].append(row['date'])
            ledger_dict['dt_exit'].append('')
            ledger_dict['success'].append('')
            ledger_dict['n'].append('')
            ledger_dict['rtn'].append('')
            enter_price.append('')
        if row['trip'] == 'ENTRY' and row['status'] == 'CANCELLED':
            ledger_dict['success'][current_tid - 1] = 0
            # get day of week and calculate n
            days = cal_days(ledger_dict['dt_enter'][current_tid - 1], row['date'])
            ledger_dict['n'][current_tid - 1] = days
        elif row['trip'] == 'ENTRY' and row['status'] == 'FILLED':
            enter_price[current_tid - 1] = row['price']
        elif row['trip'] == 'EXIT' and row['status'] == 'FILLED':
            if row['type'] == 'LMT':
                ledger_dict['success'][current_tid - 1] = 1
            ledger_dict['dt_exit'][current_tid - 1] = row['date']
            # get day of week and calculate n
            days = cal_days(ledger_dict['dt_enter'][current_tid - 1], row['date'])
            ledger_dict['n'][current_tid - 1] = days

            # return rate
            rtn = log(row['price'] / enter_price[current_tid - 1]) / days
            rtn_percentage = str(round(rtn * 100, 3)) + "%"
            ledger_dict['rtn'][current_tid - 1] = rtn_percentage
        elif row['trip'] == 'EXIT' and row['status'] == 'CANCELLED':
            ledger_dict['success'][current_tid - 1] = -1
            # n
            days = cal_days(ledger_dict['dt_enter'][current_tid - 1], row['date'])
            ledger_dict['n'][current_tid - 1] = days

        pre_trade_id = current_tid

    ledger = pd.DataFrame.from_dict(ledger_dict)
    # print(ledger.head(50))
    return ledger.to_dict("records")


def cal_days(sdate_str, edate_str):
    # format = "%m/%d/%y"
    format = "%Y-%m-%d"
    sdate = datetime.strptime(sdate_str, format).date()
    edate = datetime.strptime(edate_str, format).date()
    delta = timedelta(days=1)
    days = 0
    while sdate <= edate:
        if sdate.isoweekday() != 6 and sdate.isoweekday() != 7:
            days += 1
        sdate += delta
    return days


def cal_success_acc(y_pred_list, y_test_list):
    correct_num = 0
    total_num = 0
    for i in range(0, len(y_test_list)):
        if y_test_list[i] == 1:
            if y_pred_list[i] == y_test_list[i]:
                correct_num += 1
            total_num += 1
    acc = float(correct_num) / total_num
    return acc


def compare_return(pred_lgr, n3):
    rtn_benchmark = 1.0
    rtn_benchmark_list = []  # list of ivv return
    rtn_old_alpha = []  # list of ivv return when pred_lgr.rtn != 0, avoid the influence of much 0 in pred_lgr.rtn when calculating old alpha
    rtn_new_alpha = []  # list of ivv return when pred_lgr.pred_rtn != 0, avoid the influence of much 0 in pred_lgr.pred_rtn when calculating new alpha
    rtn = 1.0
    rtn_list = []  # list of old strategy return
    pred_rtn = 1.0
    pred_rtn_list = []  # list of new strategy return after prediction
    n = 0  # number of trades
    for i in range(n3, len(pred_lgr)):
        rtn_benchmark_list.append(pred_lgr.rtn_benchmark.iloc[i])
        if pred_lgr.rtn.iloc[i] != "":
            val = float(pred_lgr.rtn.iloc[i].strip("%"))
            rtn *= (1 + val / 100)
            rtn_list.append(val / 100)
            rtn_old_alpha.append(pred_lgr.rtn_benchmark.iloc[i])
        else:
            # rtn_list.append(0)
            pass

        if pred_lgr.pred_rtn.iloc[i] != "":
            val = float(pred_lgr.pred_rtn.iloc[i].strip("%"))
            pred_rtn *= (1 + val / 100)
            pred_rtn_list.append(val / 100)
            n += 1
            rtn_new_alpha.append(pred_lgr.rtn_benchmark.iloc[i])
        else:
            # pred_rtn_list.append(0)
            pass
    # convert total return to annualized geometric mean return
    rtn = (rtn ** (1 / len(rtn_benchmark_list)) - 1) * 255
    pred_rtn = (pred_rtn ** (1 / len(rtn_benchmark_list)) - 1) * 255

    # calculate old alpha
    # X = np.array(rtn_benchmark_list).reshape(-1, 1)
    X = np.array(rtn_old_alpha).reshape(-1, 1)
    y = np.array(rtn_list).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    old_alpha = model.intercept_[0]
    # calculate new alpha
    X = np.array(rtn_new_alpha).reshape(-1, 1)
    y = np.array(pred_rtn_list).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    new_alpha = model.intercept_[0]
    # convert alpha to annualized
    old_alpha = old_alpha / len(rtn_benchmark_list) * 255
    new_alpha = new_alpha / len(rtn_benchmark_list) * 255

    b = 0.001
    a = -0.1
    hoeffding_prob = 2 * np.exp(-2 * n * (new_alpha - old_alpha) ** 2 / (b - a) ** 2)
    # print("annualized return benchmark (Geometric Mean): {:.2f}%".format(rtn * 100))
    # print("annualized predicted return (Geometric Mean): {:.2f}%".format(pred_rtn * 100))
    # print("old alpha over benchmark: {:.4f}".format(old_alpha))
    # print("new alpha over benchmark: {:.4f}".format(new_alpha))
    # print("Hoeffding Inequality: P(new_alpha - old_alpha >= {:.4f}) <= {:.4f}".format(new_alpha - old_alpha,
    #                                                                                   hoeffding_prob))

    return rtn, pred_rtn, old_alpha, new_alpha, hoeffding_prob


@app.callback(
    Output("predict-tbl", "data"),
    Output("predict-text", "children"),
    Input("run-predict", "n_clicks"),
    [State("all_orders-tbl", 'data'), State('n-train', 'value')],
    prevent_initial_call=True
)
def predict_classifier(n_clicks, blotter_dict, n):
    feature_file = "data_cleaned.csv"
    features = pd.read_csv(feature_file)
    blotter = pd.DataFrame(blotter_dict)
    print(blotter)
    print('')
    lgr_dict = ledger(blotter)  # get the ledger
    lgr = pd.DataFrame(lgr_dict)
    # lgr['Date'] = pd.to_datetime(lgr['Date']).dt.date
    # calculate return benchmark, add in ledger
    print(lgr)
    ivv_rtn = features["IVV US Equity"].pct_change()[1:]
    lgr.loc[0, "rtn_benchmark"] = 0
    lgr.loc[1:, "rtn_benchmark"] = ivv_rtn
    # lgr.to_csv("ledger.csv")

    features_len = features.shape[0]
    # train_len = int(round(features_len/6*5,0))
    # n: the length of each training set
    train_len = n

    success = lgr['success']
    features = pd.concat([features,success],axis=1)
    y_test_list = []
    y_pred_list = []
    pred_lgr = lgr.copy()  # predicted ledger for output and comparison
    pred_lgr["pred_success"] = ""
    pred_lgr["pred_rtn"] = ""
    for i in range(train_len, features_len):
        train_start_index = i - train_len
        train_set = features.drop('Dates', axis=1).iloc[train_start_index:train_start_index + train_len]
        train_set = train_set.values
        test_set = features.drop('Dates', axis=1).iloc[train_start_index + train_len]
        test_set = test_set.values.reshape(1, -1)
        # print(test_set)
        y_train_data = np.asarray(lgr.success.iloc[train_start_index:train_start_index + train_len], dtype=int)
        y_test = lgr.success.iloc[train_start_index + train_len]
        if y_test != '':
            y_test_list.append(y_test)
            # y_test_data = np.asarray(y_test, dtype=int)
        else:
            break

        scaler = StandardScaler()
        scaler.fit(train_set)
        x_train_data = scaler.transform(train_set)
        x_test_data = scaler.transform(test_set)

        pca = PCA(0.97)
        pca.fit(x_train_data)
        x_train_data = pca.transform(x_train_data)
        x_test_data = pca.transform(x_test_data)

        classifier = Perceptron(eta0=0.1)
        classifier.fit(x_train_data, y_train_data)

        y_pred = classifier.predict(x_test_data)
        y_pred_list.append(y_pred[0])
        # add predicted results for ouput and comparison
        pred_lgr.loc[i, 'pred_success'] = y_pred[0]
        if y_pred[0] == 1:
            pred_lgr.loc[i, 'pred_rtn'] = pred_lgr.rtn.iloc[i]
        else:
            pass

    # print(pred_lgr[-50:])
    # print(y_test_list)
    # print(y_pred_list)
    acc_total = accuracy_score(y_pred_list, y_test_list)
    acc_success = cal_success_acc(y_pred_list, y_test_list)
    print("total accuracy: {:.4f}".format(acc_total))
    print("accuracy of success: {:.4f}".format(acc_success))

    rtn, pred_rtn, old_alpha, new_alpha, hoeffding_prob = compare_return(pred_lgr, n)

    print("annualized return benchmark: {:.2f}%".format(rtn*100))
    print("annualized predicted return: {:.2f}%".format(pred_rtn*100))
    print("old alpha over benchmark: {:.4f}".format(old_alpha))
    print("new alpha over benchmark: {:.4f}".format(new_alpha))
    print("Hoeffding Inequality: P(new_alpha - old_alpha >= {:.4f}) <= {:.4f}".format(new_alpha - old_alpha,
                                                                                      hoeffding_prob))
    summ = "Total accuracy: {:.4f}. \n Accuracy of success: {:.4f}. \n Annualized return benchmark: {:.2f}%. \n" \
           "Annualized predicted return: {:.2f}%. \nOld alpha over benchmark: {:.4f}. \nNew alpha over benchmark: {:.4f}. \n" \
           "Hoeffding Inequality: P(new_alpha - old_alpha >= {:.4f}) <= {:.4f}.".format(acc_total, acc_success, rtn*100,
                                                                                        pred_rtn*100, old_alpha, new_alpha,
                                                                                        new_alpha - old_alpha, hoeffding_prob)
    # return y_pred_list, pred_lgr, acc_total, acc_success, rtn, pred_rtn, old_alpha, new_alpha, hoeffding_prob
    return pred_lgr.to_dict("records"), summ


# predict function end


if __name__ == '__main__':
    app.run_server(debug=True)
