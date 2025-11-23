# -*- coding: utf-8 -*-
import dash
from dash import html, dcc, Input, Output, callback, State, dash_table, clientside_callback
import dash_mantine_components as dmc
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from scipy import stats
import numpy as np
import traceback
import warnings
warnings.filterwarnings('ignore')

est = pytz.timezone('US/Eastern')
app = dash.Dash(__name__, title="Market Voyager Terminal", suppress_callback_exceptions=True)

INDICES_DATA = {
    "S&P 500": {"country": "USA", "region": "Americas", "symbol": "^GSPC", "flag": "🇺🇸"},
    "Dow Jones": {"country": "USA", "region": "Americas", "symbol": "^DJI", "flag": "🇺🇸"},
    "NASDAQ": {"country": "USA", "region": "Americas", "symbol": "^IXIC", "flag": "🇺🇸"},
    "TSX": {"country": "Canada", "region": "Americas", "symbol": "^GSPTSE", "flag": "🇨🇦"},
    "FTSE 100": {"country": "UK", "region": "Europe", "symbol": "^FTSE", "flag": "🇬🇧"},
    "DAX": {"country": "Germany", "region": "Europe", "symbol": "^GDAXI", "flag": "🇩🇪"},
    "CAC 40": {"country": "France", "region": "Europe", "symbol": "^FCHI", "flag": "🇫🇷"},
    "Nikkei 225": {"country": "Japan", "region": "Asia", "symbol": "^N225", "flag": "🇯🇵"},
    "Hang Seng": {"country": "Hong Kong", "region": "Asia", "symbol": "^HSI", "flag": "🇭🇰"},
    "SSE": {"country": "China", "region": "Asia", "symbol": "000001.SS", "flag": "🇨🇳"},
    "Sensex": {"country": "India", "region": "Asia", "symbol": "^BSESN", "flag": "🇮🇳"},
    "ASX 200": {"country": "Australia", "region": "Asia", "symbol": "^AXJO", "flag": "🇦🇺"},
}

INDEX_CONSTITUENTS = {
    "S&P 500": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "LLY", "AVGO"],
    "Dow Jones": ["AAPL", "MSFT", "UNH", "GS", "HD", "MCD", "CAT", "AMGN", "V", "BA"],
    "NASDAQ": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST", "NFLX"],
    "TSX": ["RY.TO", "TD.TO", "BNS.TO", "ENB.TO", "CNQ.TO", "CNR.TO", "CP.TO", "SU.TO"],
    "FTSE 100": ["SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "DGE.L", "RIO.L"],
    "DAX": ["SAP.DE", "SIE.DE", "ALV.DE", "AIR.DE", "DTE.DE", "VOW3.DE", "MBG.DE", "BAS.DE"],
    "CAC 40": ["MC.PA", "OR.PA", "SAN.PA", "TTE.PA", "AIR.PA", "BNP.PA", "SU.PA", "CS.PA"],
    "Nikkei 225": ["7203.T", "6758.T", "9984.T", "9433.T", "6861.T", "8306.T", "8035.T", "6367.T"],
    "Hang Seng": ["0700.HK", "9988.HK", "0941.HK", "0005.HK", "0388.HK", "1299.HK", "2318.HK"],
    "SSE": ["600519.SS", "601318.SS", "600036.SS", "600276.SS", "600887.SS", "601012.SS"],
    "Sensex": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "BHARTIARTL.NS"],
    "ASX 200": ["BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "ANZ.AX", "WES.AX"],
}

CRISIS_PERIODS = {
    "2008 Crisis": {"start": "2008-09-15", "end": "2009-03-09"},
    "COVID-19": {"start": "2020-02-20", "end": "2020-03-23"},
    "Dot-com": {"start": "2000-03-10", "end": "2002-10-09"},
    "EU Debt": {"start": "2010-04-27", "end": "2012-07-26"},
    "2022": {"start": "2022-01-01", "end": "2022-10-13"},
}

DATA_CACHE = {}
STOCK_CACHE = {}

def get_data(symbol):
    try:
        hist = yf.download(symbol, start=datetime.now() - timedelta(days=25*365), 
                          end=datetime.now(), progress=False)
        if not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            price = float(hist['Close'].iloc[-1])
            DATA_CACHE[symbol] = (price, hist)
            return price, hist
    except: pass
    if symbol in DATA_CACHE: return DATA_CACHE[symbol]
    return None, pd.DataFrame()

def get_stock_performance(ticker, period="5d"):
    try:
        # Map period to yfinance valid periods if needed, or fetch enough data
        # yf.download periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        yf_period = period
        if period == "1d": yf_period = "5d" # Need prev close for 1d
        elif period == "intraday": yf_period = "1d" # Intraday needs 1d (but we might need 1m interval for true intraday, here we stick to daily close for simplicity or 5d for safety)
        
        # For simplicity and cache efficiency, we might just fetch "1y" and slice?
        # But yfinance is faster with smaller chunks. Let's stick to requested period.
        
        # Adjust for "intraday" request -> usually means "today's move".
        if period == "Intraday %":
            hist = yf.download(ticker, period="5d", progress=False) # Get recent data
            if not hist.empty:
                if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
                current = hist['Close'].iloc[-1]
                open_p = hist['Open'].iloc[-1]
                # Check date match for true intraday
                if hist.index[-1].date() != datetime.now(est).date():
                    change = 0.0 # Market closed
                else:
                    change = ((current - open_p) / open_p) * 100
                return {"ticker": ticker, "price": float(current), "change": round(float(change), 2)}
                
        # Map column names to days/periods
        days_map = {
            "1D %": 2, "5D %": 5, "1M %": 21, "YTD %": "ytd", 
            "1Y %": 252, "3Y %": 756
        }
        
        target = days_map.get(period, 5) # Default to 5 days
        
        if target == "ytd":
            hist = yf.download(ticker, period="ytd", progress=False)
        elif isinstance(target, int):
            # Fetch a bit more to ensure we have the start point
            fetch_period = "5y" if target > 252 else "2y" if target > 21 else "3mo" if target > 5 else "1mo"
            hist = yf.download(ticker, period=fetch_period, progress=False)
        else:
            hist = yf.download(ticker, period="5d", progress=False)
            
        if not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
            
            current = hist['Close'].iloc[-1]
            
            if target == "ytd":
                start_price = hist['Close'].iloc[0]
            elif isinstance(target, int):
                if len(hist) >= target:
                    start_price = hist['Close'].iloc[-target] # Approximate N trading days ago
                else:
                    start_price = hist['Close'].iloc[0]
            else:
                start_price = hist['Close'].iloc[0]
                
            change = ((current - start_price) / start_price) * 100
            return {"ticker": ticker, "price": float(current), "change": round(float(change), 2)}
            
    except Exception as e: 
        # print(f"Error fetching {ticker}: {e}")
        pass
    return None

def get_top_movers(index_name, period="1D %"):
    constituents = INDEX_CONSTITUENTS.get(index_name, [])
    if not constituents: return [], []
    performances = []
    for ticker in constituents:
        perf = get_stock_performance(ticker, period)
        if perf: performances.append(perf)
    if not performances: return [], []
    
    # Sort by change
    sorted_perf = sorted(performances, key=lambda x: x['change'], reverse=True)
    return sorted_perf[:5], sorted_perf[-5:][::-1], performances # Return full list too for tape

def calc_sigma(hist, current_move):
    try:
        if len(hist) < 30 or current_move is None: return None
        daily_returns = hist['Close'].pct_change().dropna() * 100
        std_dev = daily_returns.std()
        mean = daily_returns.mean()
        if std_dev == 0: return 0
        sigma = (current_move - mean) / std_dev
        return round(sigma, 2)
    except: return None

def calc_live_1d(hist):
    try:
        if len(hist) < 2: return None
        current = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        return round(((current - prev_close) / prev_close) * 100, 2)
    except: return None

def calc_intraday(hist):
    try:
        if len(hist) < 1: return None
        
        # Check if the last data point is from today
        last_date = hist.index[-1].date()
        current_date = datetime.now(est).date()
        
        if last_date != current_date:
            return 0.00 # Market closed/Holiday
            
        current = hist['Close'].iloc[-1]
        open_price = hist['Open'].iloc[-1]
        return round(((current - open_price) / open_price) * 100, 2)
    except: return None

def calc(hist, days):
    try:
        if len(hist) < 2: return None
        cur, old = hist['Close'].iloc[-1], hist['Close'].iloc[-days-1 if len(hist) > days else 0]
        return round(((cur - old) / old) * 100, 2) if old > 0 else None
    except: return None

def crisis_perf(hist, start, end):
    try:
        c = hist[(hist.index >= pd.to_datetime(start)) & (hist.index <= pd.to_datetime(end))]
        if len(c) >= 2:
            return round(((c['Close'].iloc[-1] - c['Close'].iloc[0]) / c['Close'].iloc[0]) * 100, 2)
    except: pass
    return None

def create_chart(hist, name, start_date=None, end_date=None, title_suffix=""):
    if start_date and end_date:
        mask = (hist.index >= pd.to_datetime(start_date)) & (hist.index <= pd.to_datetime(end_date))
        plot_data = hist.loc[mask]
    elif start_date:
        mask = (hist.index >= pd.to_datetime(start_date))
        plot_data = hist.loc[mask]
    else:
        plot_data = hist

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_data.index, y=plot_data['Close'], name=name,
        line=dict(color='#339af0', width=2),
        fill='tozeroy', fillcolor='rgba(51, 154, 240, 0.1)'
    ))
    
    if start_date is None:
        for crisis, dates in CRISIS_PERIODS.items():
            c_start, c_end = pd.to_datetime(dates['start']), pd.to_datetime(dates['end'])
            c = plot_data[(plot_data.index >= c_start) & (plot_data.index <= c_end)]
            if not c.empty:
                fig.add_vrect(x0=c_start, x1=c_end, fillcolor="red", opacity=0.1, 
                             layer="below", line_width=0, annotation_text=crisis,
                             annotation_position="top left", annotation_font_size=9)
    
    fig.update_layout(
        template='plotly_dark', height=350,
        margin=dict(l=40, r=20, t=40, b=30),
        title=f"{name} - {title_suffix}" if title_suffix else f"{name} - Historical Performance",
        xaxis_title=None, yaxis_title=None,
        hovermode='x unified', showlegend=False
    )
    return fig

def get_news(index_name):
    return [
        {"title": f"Breaking: {index_name} sees significant volatility amid global uncertainty.", "source": "Bloomberg", "time": "1h ago"},
        {"title": f"Market Update: Tech sector is leading {index_name} movements today.", "source": "Reuters", "time": "2h ago"},
        {"title": f"Analysis: Experts revise outlook for key {index_name} companies.", "source": "CNBC", "time": "4h ago"},
        {"title": f"Economy: Key data release impacts {index_name} trading volume.", "source": "WSJ", "time": "6h ago"},
        {"title": f"Technical: {index_name} approaches critical support levels.", "source": "Financial Times", "time": "8h ago"},
    ]

def get_sigma_styles():
    styles = []
    styles.append({'if': {'filter_query': '{1D Sigma} >= 0.5 && {1D Sigma} < 1.0', 'column_id': '1D Sigma'}, 'color': '#8ce99a', 'fontWeight': 'bold'})
    styles.append({'if': {'filter_query': '{1D Sigma} >= 1.0 && {1D Sigma} < 2.0', 'column_id': '1D Sigma'}, 'color': '#51cf66', 'fontWeight': 'bold'})
    styles.append({'if': {'filter_query': '{1D Sigma} >= 2.0 && {1D Sigma} < 3.0', 'column_id': '1D Sigma'}, 'color': '#2f9e44', 'fontWeight': 'bold'})
    styles.append({'if': {'filter_query': '{1D Sigma} >= 3.0', 'column_id': '1D Sigma'}, 'color': '#2b8a3e', 'fontWeight': 'bold', 'textDecoration': 'underline'})
    styles.append({'if': {'filter_query': '{1D Sigma} <= -0.5 && {1D Sigma} > -1.0', 'column_id': '1D Sigma'}, 'color': '#ff8787', 'fontWeight': 'bold'})
    styles.append({'if': {'filter_query': '{1D Sigma} <= -1.0 && {1D Sigma} > -2.0', 'column_id': '1D Sigma'}, 'color': '#fa5252', 'fontWeight': 'bold'})
    styles.append({'if': {'filter_query': '{1D Sigma} <= -2.0 && {1D Sigma} > -3.0', 'column_id': '1D Sigma'}, 'color': '#e03131', 'fontWeight': 'bold'})
    styles.append({'if': {'filter_query': '{1D Sigma} <= -3.0', 'column_id': '1D Sigma'}, 'color': '#c92a2a', 'fontWeight': 'bold', 'textDecoration': 'underline'})
    return styles

app.layout = dmc.MantineProvider([
    dmc.Container([
        # Header
        dmc.Group([
            dmc.Stack([
                dmc.Group([
                    dmc.ThemeIcon(
                        dmc.Text("🚀", size="xl"),
                        size="xl", radius="xl", variant="light", color="blue"
                    ),
                    dmc.Title("Market Voyager Terminal", order=2),
                ], gap="xs"),
                dmc.Group([
                    dmc.Text(id="timestamp", size="sm", c="dimmed"),
                    dmc.Badge(id="market-status", variant="filled"), # Dynamic Status
                ], gap="xs"),
            ], gap=0),
            
            dmc.Group([
                dmc.Button("Refresh", id="refresh", leftSection="🔄", variant="light", size="sm"),
                dmc.TextInput(id="search", placeholder="Search...", w=200, leftSection="🔍"),
            ]),
        ], justify="space-between", mb="md", mt="md"),
        
        # Split View Layout with Resizable Left Panel
        html.Div([
            # Left Panel: Table + News
            html.Div([
                # Table: Auto height, no scroll
                html.Div(id="table-container", style={"marginBottom": "10px"}),
                
                # News: Scrollable
                dmc.Paper(id="news-container", p="xs", shadow="sm", withBorder=True, 
                         style={"height": "300px", "overflowY": "auto", "backgroundColor": "#1A1B1E"})
            ], style={
                "width": "60%", "height": "calc(100vh - 150px)", 
                "resize": "horizontal", "overflow": "auto", 
                "paddingRight": "10px", "minWidth": "300px", "maxWidth": "90%",
                "borderRight": "1px solid #373a40" # Visual cue
            }),
            
            # Right Panel: Details
            html.Div([
                dmc.Paper(id="detail-panel", p="md", shadow="md", withBorder=True, 
                         style={"height": "100%", "overflowY": "auto"}),
                dcc.Loading(id="loading", type="default", children=html.Div(id="loading-output")),
            ], style={
                "flex": "1", "height": "calc(100vh - 150px)", 
                "marginLeft": "10px", "overflow": "hidden"
            }),
        ], style={"display": "flex", "width": "100%", "height": "100%"}),
        
        # Ticker Tape Footer
        html.Div(id="ticker-tape-container", style={
            "position": "fixed", "bottom": "0", "left": "0", "width": "100%", 
            "backgroundColor": "#1a1b1e", "borderTop": "1px solid #373a40", 
            "padding": "8px 0", "zIndex": "1000", "overflow": "hidden", "whiteSpace": "nowrap"
        }),
        
        dcc.Store(id="selected-index", data="S&P 500"),
        dcc.Store(id="speech-text"), # Store text to speak
        dcc.Store(id="tts-dummy-output"), # Dummy output for client-side callback
        dcc.Interval(id="interval-component", interval=15*1000, n_intervals=0),
        
    ], size="100%", px="md", style={"height": "100vh", "overflow": "hidden"}),
], defaultColorScheme="dark")

# Client-side callback for Text-to-Speech
app.clientside_callback(
    """
    function(n_clicks, text) {
        if (n_clicks > 0 && text) {
            var msg = new SpeechSynthesisUtterance();
            msg.text = text;
            msg.lang = 'en-US';
            msg.rate = 1.0;
            msg.pitch = 1.0;
            window.speechSynthesis.cancel(); // Stop previous
            window.speechSynthesis.speak(msg);
        }
        return "";
    }
    """,
    Output("tts-dummy-output", "data"), # Fixed: Output to dummy store
    Input("speak-btn", "n_clicks"),
    State("speech-text", "data"),
    prevent_initial_call=True
)

@callback(
    [Output("timestamp", "children"),
     Output("table-container", "children"),
     Output("market-status", "children"),
     Output("market-status", "color"),
     Output("market-status", "className")],
    [Input("refresh", "n_clicks"),
     Input("interval-component", "n_intervals"),
     Input("search", "value")],
    prevent_initial_call=False,
)
def update_table(refresh, n_intervals, search):
    try:
        now = datetime.now(est).strftime('%Y-%m-%d %I:%M:%S %p EST')
        current_date = datetime.now(est).date()
        
        # Determine Market Status based on S&P 500 data date
        market_open = False
        sp500_sym = INDICES_DATA["S&P 500"]["symbol"]
        _, sp500_hist = get_data(sp500_sym)
        if not sp500_hist.empty:
            last_date = sp500_hist.index[-1].date()
            if last_date == current_date:
                market_open = True
        
        status_text = "MARKET OPEN" if market_open else "MARKET CLOSED"
        status_color = "green" if market_open else "gray"
        status_class = "blink-me" if market_open else ""
        
        rows = []
        for name, info in INDICES_DATA.items():
            if search and search.lower() not in name.lower() and search.lower() not in info["country"].lower():
                continue
                
            price, hist = get_data(info["symbol"])
            if price and not hist.empty:
                live_1d = calc_live_1d(hist)
                intraday = calc_intraday(hist)
                sigma = calc_sigma(hist, live_1d)
                
                row = {
                    "Index": name,
                    "Country": f"{info['flag']} {info['country']}",
                    "Price": f"${price:,.2f}",
                    "Intraday %": intraday,
                    "1D %": live_1d,
                    "1D Sigma": sigma,
                    "5D %": calc(hist, 5), # Renamed
                    "1M %": calc(hist, 21),
                    "YTD %": calc(hist, 252),
                    "1Y %": calc(hist, 252),
                    "3Y %": calc(hist, 756),
                }
                
                for crisis in CRISIS_PERIODS.keys():
                    row[crisis] = crisis_perf(hist, CRISIS_PERIODS[crisis]["start"], CRISIS_PERIODS[crisis]["end"])
                
                rows.append(row)
        
        if not rows:
            return f"Last Updated: {now}", dmc.Alert("No data available", color="yellow"), status_text, status_color, status_class
        
        df = pd.DataFrame(rows)
        
        style_data_conditional = [
            {'if': {'column_id': 'Index'}, 'fontWeight': 'bold', 'color': '#339af0', 'cursor': 'pointer'},
            {'if': {'column_id': 'Price'}, 'color': '#4dabf7', 'fontWeight': '600'},
            *[{'if': {'filter_query': f'{{{col}}} > 0', 'column_id': col}, 'color': '#40c057', 'fontWeight': 'bold'} 
              for col in ['1D %', 'Intraday %', '5D %', '1M %', 'YTD %', '1Y %', '3Y %'] + list(CRISIS_PERIODS.keys())],
            *[{'if': {'filter_query': f'{{{col}}} < 0', 'column_id': col}, 'color': '#fa5252', 'fontWeight': 'bold'} 
              for col in ['1D %', 'Intraday %', '5D %', '1M %', 'YTD %', '1Y %', '3Y %'] + list(CRISIS_PERIODS.keys())],
        ]
        style_data_conditional.extend(get_sigma_styles())
        
        table = dash_table.DataTable(
            id='indices-table',
            columns=[
                {'name': 'Index', 'id': 'Index', 'type': 'text'},
                {'name': 'Country', 'id': 'Country', 'type': 'text'},
                {'name': 'Price', 'id': 'Price', 'type': 'text'},
                {'name': 'Intraday %', 'id': 'Intraday %', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
                {'name': '1D %', 'id': '1D %', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
                {'name': '1D Sigma', 'id': '1D Sigma', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
                {'name': '5D %', 'id': '5D %', 'type': 'numeric', 'format': {'specifier': '+.2f'}}, # Renamed
                {'name': '1M %', 'id': '1M %', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
                {'name': 'YTD %', 'id': 'YTD %', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
                {'name': '1Y %', 'id': '1Y %', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
                {'name': '3Y %', 'id': '3Y %', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
                *[{'name': c, 'id': c, 'type': 'numeric', 'format': {'specifier': '+.2f'}} for c in CRISIS_PERIODS.keys()],
            ],
            data=df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#1a1b1e',
                'color': '#c1c2c5', 'border': '1px solid #373a40', 'fontSize': '13px',
            },
            style_header={
                'backgroundColor': '#25262b', 'fontWeight': 'bold', 'border': '1px solid #373a40',
                'color': '#909296', 'fontSize': '12px',
            },
            style_data_conditional=style_data_conditional,
            sort_action='native', filter_action='native', page_size=20, style_as_list_view=True,
            active_cell={'row': 0, 'column': 0, 'column_id': 'Index'},
        )
        
        return f"Last Updated: {now}", table, status_text, status_color, status_class
    except Exception as e:
        print(f"Error in update_table: {e}")
        traceback.print_exc()
        return "Error", dmc.Alert(f"Application Error: {str(e)}", color="red"), "", "", ""

@callback(
    [Output("detail-panel", "children"),
     Output("news-container", "children"),
     Output("loading-output", "children"),
     Output("ticker-tape-container", "children"),
     Output("speech-text", "data")],
    [Input('indices-table', 'active_cell'),
     State('indices-table', 'data')],
)
def update_detail(active_cell, data):
    try:
        # Default return values
        empty_return = (dmc.Alert("Select an index to view details", color="blue"), None, None, None, "")
        
        if not data:
            return empty_return
            
        if not active_cell:
            # Default to first row if available
            row = data[0]
            col_id = "Index"
            index_name = row['Index']
        else:
            row = data[active_cell['row']]
            col_id = active_cell['column_id']
            index_name = row['Index']

        info = INDICES_DATA.get(index_name)
        if not info: return empty_return
        
        price, hist = get_data(info["symbol"])
        if hist.empty: return dmc.Alert("Error loading data", color="red"), None, None, None, ""
        
        # Determine Period from Clicked Column
        period_map = {
            "Intraday %": "Intraday %",
            "1D %": "1D %",
            "5D %": "5D %",
            "1M %": "1M %",
            "YTD %": "YTD %",
            "1Y %": "1Y %",
            "3Y %": "3Y %",
            "Price": "Max" # Price click -> Full History
        }
        
        # Default to Intraday (Live Chart) if no specific column or initial load
        selected_period = period_map.get(col_id, "Intraday %") 
        
        # Fetch Top Movers for Selected Period
        # For "Max", we probably just want 1D movers or maybe 1Y? Let's stick to 1D for relevance.
        movers_period = selected_period if selected_period != "Max" else "1D %"
        winners, losers, all_perfs = get_top_movers(index_name, movers_period)
        
        # Generate Ticker Tape Items (Sorted by absolute move in this period)
        sorted_constituents = sorted(all_perfs, key=lambda x: abs(x['change']), reverse=True)
        tape_items = []
        for perf in sorted_constituents:
            color = "#40c057" if perf['change'] >= 0 else "#fa5252"
            tape_items.append(
                html.Span([
                    html.Span(f"{perf['ticker']} ", style={"fontWeight": "bold", "color": "#fff"}),
                    html.Span(f"${perf['price']:.2f} ", style={"color": "#adb5bd"}),
                    html.Span(f"({perf['change']:+.2f}%)", style={"color": color, "fontWeight": "bold"}),
                    html.Span("  -  ", style={"color": "#555", "margin": "0 15px"})
                ], style={"display": "inline-block"})
            )
            
        if not tape_items:
            tape_items = [html.Span("Loading constituent data...", style={"color": "#adb5bd", "padding": "0 20px"})]
                
        ticker_tape = html.Div(tape_items, style={
            "display": "inline-block",
            "animation": "marquee 30s linear infinite",
            "whiteSpace": "nowrap"
        })
        
        ticker_tape_wrapper = html.Div([ticker_tape])

        # Generate News Component
        news_items = get_news(index_name)
        news_narrative = " In the news: "
        for item in news_items:
            news_narrative += f"From {item['source']}: {item['title']} "
            
        if winners:
            winner_text = f"Top movers for {selected_period} include {winners[0]['ticker']} up {winners[0]['change']} percent."
        if losers:
            loser_text = f"On the downside, {losers[0]['ticker']} is down {abs(losers[0]['change'])} percent."
        
        speech_text = (
            f"Update for {index_name}. "
            f"Currently at {row['Price']}. "
            f"Showing data for {selected_period}. "
            f"{winner_text} {loser_text} "
            f"{news_narrative}"
        )
        
        # ... (Avatar/News UI code remains same) ...
        avatar_url = "https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/avatars/avatar-1.png"
        news_component = dmc.Stack([
            dmc.Group([
                dmc.Group([
                        "backgroundColor": "#2C2E33", "borderTopLeftRadius": "0", "position": "relative"
                    }) for n in news_items[:3]]
                ], style={"flex": 1}, gap="xs")
            ], align="start", gap="md")
        ], gap="xs", p="xs")

        # Chart Logic for Selected Period
        chart_days_map = {
            "Intraday %": 1, "1D %": 1, "5D %": 5, "1M %": 30, 
            "YTD %": "ytd", "1Y %": 365, "3Y %": 365*3
        }
        
        if col_id in CRISIS_PERIODS:
            dates = CRISIS_PERIODS[col_id]
            chart = create_chart(hist, index_name, dates['start'], dates['end'], f"{col_id}")
            chart_title = f"{index_name} - {col_id}"
        else:
            days = chart_days_map.get(selected_period, 365)
            
            # Special handling for Intraday/1D to show timestamps
            if selected_period in ["Intraday %", "1D %"]:
                # Fetch high-frequency data
                try:
                    # 1d period with 5m interval
                    intra_hist = yf.download(info["symbol"], period="1d" if selected_period == "Intraday %" else "5d", interval="5m", progress=False)
                    if not intra_hist.empty:
                        if isinstance(intra_hist.columns, pd.MultiIndex): 
                            intra_hist.columns = intra_hist.columns.get_level_values(0)
                        
                        # If 1D %, we might want just the last 2 days? 
                        # "1D %" usually implies "Today" or "Last Session".
                        # If we want strictly 1 day, period="1d".
                        # If we want context, maybe 2d?
                        # Let's stick to "1d" for Intraday and "5d" (zoomed to last 2 days) for 1D?
                        # Actually, standard is: Intraday = Today. 1D = Today (same chart usually).
                        # But user asked for timestamps.
                        
                        if selected_period == "1D %":
                             # Show last 2 days to see the gap? Or just today?
                             # Usually "1D" chart is just today.
                             intra_hist = intra_hist # Use the fetched data
                        
                        chart = create_chart(intra_hist, index_name, title_suffix=selected_period)
                        chart.update_xaxes(tickformat="%H:%M") # Show Time
                    else:
                        # Fallback to daily if intraday fails
                        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                        chart = create_chart(hist, index_name, start_date=start_date, title_suffix=selected_period)
                except:
                    start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                    chart = create_chart(hist, index_name, start_date=start_date, title_suffix=selected_period)
            
            elif selected_period == "Max":
                # Full History
                chart = create_chart(hist, index_name, title_suffix="All Time")
                chart_title = f"{index_name} - All Time"
            
            elif days == "ytd":
                start_date = datetime(datetime.now().year, 1, 1).strftime('%Y-%m-%d')
                chart = create_chart(hist, index_name, start_date=start_date, title_suffix="YTD")
                chart_title = f"{index_name} - YTD"
            else:
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                chart = create_chart(hist, index_name, start_date=start_date, title_suffix=selected_period)
                chart_title = f"{index_name} - {selected_period}"

        winner_cards = [dmc.Card([
            dmc.Group([dmc.Text(s['ticker'], fw=700, size="sm"), dmc.Badge(f"+{s['change']:.2f}%", color="green", size="sm")], justify="space-between"),
            dmc.Text(f"${s['price']:.2f}", size="xs", c="dimmed")
        ], p="xs", withBorder=True, mb="xs") for s in winners]
        
        loser_cards = [dmc.Card([
            dmc.Group([dmc.Text(s['ticker'], fw=700, size="sm"), dmc.Badge(f"{s['change']:.2f}%", color="red", size="sm")], justify="space-between"),
            dmc.Text(f"${s['price']:.2f}", size="xs", c="dimmed")
        ], p="xs", withBorder=True, mb="xs") for s in losers]
        
        detail = [
            dmc.Group([
                dmc.Stack([
                    dmc.Title(f"{info['flag']} {index_name}", order=3),
                    dmc.Group([
                        dmc.Text(f"{selected_period}: {row.get(selected_period, 0):+.2f}%" if isinstance(row.get(selected_period), (int, float)) else f"{selected_period}", size="sm", fw=500),
                        dmc.Badge(f"{row['1D Sigma']}σ", color="yellow", variant="outline", size="sm")
                    ], gap="xs")
                ], gap=0),
                dmc.Text(row['Price'], size="xl", fw=700, c="blue"),
            ], justify="space-between", mb="md"),
            
            dcc.Graph(figure=chart, config={'displayModeBar': False}, style={"height": "300px"}),
            
            dmc.Grid([
                dmc.GridCol([
                    dmc.Text(f"Top Gainers ({selected_period})", c="green", fw=700, size="sm", mb="xs"),
                    *winner_cards
                ], span=6),
                dmc.GridCol([
                    dmc.Text(f"Top Losers ({selected_period})", c="red", fw=700, size="sm", mb="xs"),
                    *loser_cards
                ], span=6),
            ], gutter="xs", mt="md"),
        ]
        return detail, news_component, None, ticker_tape_wrapper, speech_text
    except Exception as e:
        print(f"Error in update_detail: {e}")
        traceback.print_exc()
        # Ensure exactly 5 outputs are returned even on error
        return dmc.Alert(f"Error: {str(e)}", color="red"), None, None, None, ""

if __name__ == "__main__":
    app.run(debug=True, port=8051)
