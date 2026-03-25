import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def leitura_e_saneamento():
    files = glob.glob("football_data/*.csv")
    
    dfs = []
    
    for f in files:
    
        temp = pd.read_csv(
            f,
            engine="python",
            sep=",",
            encoding="latin1",
            on_bad_lines="skip"
        )
    
        dfs.append(temp)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # 1) remover colunas Unnamed
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    
    # 2) corrigir BOM/coluna duplicada do Div
    if "ï»¿Div" in df.columns and "Div" in df.columns:
        # preenche Div com ï»¿Div quando Div estiver vazio
        df["Div"] = df["Div"].fillna(df["ï»¿Div"])
        df = df.drop(columns=["ï»¿Div"])
    elif "ï»¿Div" in df.columns and "Div" not in df.columns:
        df = df.rename(columns={"ï»¿Div": "Div"})
        
    # datas
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    
    # Time às vezes vem vazio; cria datetime só quando der
    if "Time" in df.columns:
        df["Time"] = df["Time"].astype(str).str.strip()
        df.loc[df["Time"].isin(["", "nan", "NaN", "None"]), "Time"] = np.nan
        df["kickoff_dt"] = pd.to_datetime(
            df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].fillna("00:00"),
            errors="coerce"
        )
    else:
        df["kickoff_dt"] = df["Date"]
        
        
    df_base=df.copy()
    num_cols = ["FTHG","FTAG","HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","HR","AR"]
    num_cols = [c for c in num_cols if c in df_base.columns]
    
    for c in num_cols:
        df_base[c] = pd.to_numeric(df_base[c], errors="coerce")
    
    df_base = df_base.dropna(subset=["HC","AC","HS","AS"])
    df_base = df_base[df_base["HC"].between(0, 30) & df_base["AC"].between(0, 30)]
    
    df_base["total_corners"] = df_base["HC"] + df_base["AC"]
    df_base["total_shots"] = df_base["HS"] + df_base["AS"]
    df_base["total_shots_target"] = df_base["HST"] + df_base["AST"]
    
    df_base["over_8_5"]  = (df_base["total_corners"] > 8).astype(int)
    df_base["over_9_5"]  = (df_base["total_corners"] > 9).astype(int)
    df_base["over_10_5"] = (df_base["total_corners"] > 10).astype(int)
    
    
    print(list(df_base.columns))
    print(df_base.shape)
    print(df_base.head())
    
    return df_base

def add_rolling_mean(
    df: pd.DataFrame,
    group_col: str,
    sort_col: str,
    value_col: str,
    window: int,
    out_col: str
) -> pd.DataFrame:
    """
    Cria média móvel do value_col por grupo, ordenado por sort_col,
    usando shift(1) para evitar data leakage.
    """
    # df = df.sort_values([group_col, sort_col]).copy()
    df[out_col] = (
        df.groupby(group_col, sort=False)[value_col]
          .transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
    )
    return df


def build_home_only_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:

    base = df[["Date","HomeTeam","HC","AC","HS","AS"]].copy()

    for w in windows:
        base = add_rolling_mean(base, "HomeTeam", "Date", "HC", w, f"home_avg_HC_{w}")
        base = add_rolling_mean(base, "HomeTeam", "Date", "AC", w, f"home_avg_AC_{w}")
        base = add_rolling_mean(base, "HomeTeam", "Date", "HS", w, f"home_avg_HS_{w}")
        base = add_rolling_mean(base, "HomeTeam", "Date", "AS", w, f"home_avg_AS_{w}")

    feat_cols = ["Date","HomeTeam"] + [
        f"home_avg_{stat}_{w}"
        for w in windows
        for stat in ["HC","AC","HS","AS"]
    ]

    return base[feat_cols]

def build_away_only_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:

    base = df[["Date","AwayTeam","HC","AC","HS","AS"]].copy()

    for w in windows:
        base = add_rolling_mean(base, "AwayTeam", "Date", "AC", w, f"away_avg_AC_{w}")
        base = add_rolling_mean(base, "AwayTeam", "Date", "HC", w, f"away_avg_HC_{w}")
        base = add_rolling_mean(base, "AwayTeam", "Date", "AS", w, f"away_avg_AS_{w}")
        base = add_rolling_mean(base, "AwayTeam", "Date", "HS", w, f"away_avg_HS_{w}")

    feat_cols = ["Date","AwayTeam"] + [
        f"away_avg_{stat}_{w}"
        for w in windows
        for stat in ["AC","HC","AS","HS"]
    ]

    return base[feat_cols]

def build_overall_team_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:

    home = df[["Date","HomeTeam","AwayTeam","HC","AC","HS","AS"]].copy()
    home["team"] = home["HomeTeam"]
    home["corners_for"] = home["HC"]
    home["corners_against"] = home["AC"]
    home["shots_for"] = home["HS"]
    home["shots_against"] = home["AS"]
    home = home[["Date","team","corners_for","corners_against","shots_for","shots_against"]]

    away = df[["Date","HomeTeam","AwayTeam","HC","AC","HS","AS"]].copy()
    away["team"] = away["AwayTeam"]
    away["corners_for"] = away["AC"]
    away["corners_against"] = away["HC"]
    away["shots_for"] = away["AS"]
    away["shots_against"] = away["HS"]
    away = away[["Date","team","corners_for","corners_against","shots_for","shots_against"]]

    long_df = pd.concat([home, away], ignore_index=True).sort_values(["team","Date"])

    for w in windows:
        long_df = add_rolling_mean(long_df, "team", "Date", "corners_for", w, f"avg_corners_for_{w}")
        long_df = add_rolling_mean(long_df, "team", "Date", "corners_against", w, f"avg_corners_against_{w}")
        long_df = add_rolling_mean(long_df, "team", "Date", "shots_for", w, f"avg_shots_for_{w}")
        long_df = add_rolling_mean(long_df, "team", "Date", "shots_against", w, f"avg_shots_against_{w}")

    feat_cols = ["Date","team"] + [
        f"avg_{stat}_{w}"
        for w in windows
        for stat in ["corners_for","corners_against","shots_for","shots_against"]
    ]

    return long_df[feat_cols]

def build_model_dataset(df_base: pd.DataFrame, windows: list[int] = [3,5,10,100]) -> pd.DataFrame:

    df = df_base.sort_values("Date").copy()

    # gerar features
    home_feats = build_home_only_features(df, windows)
    away_feats = build_away_only_features(df, windows)
    overall = build_overall_team_features(df, windows)

    # merge home
    df = df.merge(home_feats, on=["Date","HomeTeam"], how="left")

    # merge away
    df = df.merge(away_feats, on=["Date","AwayTeam"], how="left")

    # merge overall home
    df = df.merge(
        overall.rename(columns={"team": "HomeTeam"}),
        on=["Date","HomeTeam"],
        how="left"
    )

    overall_home = overall.rename(columns={
    "team": "HomeTeam",
    })
    
    for w in windows:
        overall_home = overall_home.rename(columns={
            f"avg_corners_for_{w}": f"home_overall_avg_corners_for_{w}",
            f"avg_corners_against_{w}": f"home_overall_avg_corners_against_{w}",
            f"avg_shots_for_{w}": f"home_overall_avg_shots_for_{w}",
            f"avg_shots_against_{w}": f"home_overall_avg_shots_against_{w}",
        })
    
    df = df.merge(overall_home, on=["Date","HomeTeam"], how="left")
    
    overall_away = overall.rename(columns={
    "team": "AwayTeam",
    })
    
    for w in windows:
        overall_away = overall_away.rename(columns={
            f"avg_corners_for_{w}": f"away_overall_avg_corners_for_{w}",
            f"avg_corners_against_{w}": f"away_overall_avg_corners_against_{w}",
            f"avg_shots_for_{w}": f"away_overall_avg_shots_for_{w}",
            f"avg_shots_against_{w}": f"away_overall_avg_shots_against_{w}",
        })
    
    df = df.merge(overall_away, on=["Date","AwayTeam"], how="left")

    # remover jogos sem histórico suficiente
    feature_cols = []

    for w in windows:

        feature_cols += [
            f"home_avg_HC_{w}",
            f"home_avg_AC_{w}",
            f"home_avg_HS_{w}",
            f"home_avg_AS_{w}",

            f"away_avg_AC_{w}",
            f"away_avg_HC_{w}",
            f"away_avg_AS_{w}",
            f"away_avg_HS_{w}",

            f"avg_corners_for_{w}",
            f"avg_corners_against_{w}",
            f"avg_shots_for_{w}",
            f"avg_shots_against_{w}",
        ]

    df = df.dropna(subset=feature_cols)

    return df


def separacao(df):

    target = "over_9_5"

    X = df.drop(columns=[
        "over_8_5",
        "over_9_5",
        "over_10_5",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "Div",
        "Time",
        "kickoff_dt",
        #LEAKAGE
        'total_corners','HC','AC',
        # estatísticas do jogo (só existem após o apito final)
        "HS", "AS", "HST", "AST", "HF", "AF", "HY", "AY", "HR", "AR",
        "FTHG", "FTAG", "FTR",
        "HTHG", "HTAG", "HTR",
        "HO", "AO",  # chutes no alvo (game stats)
        "total_shots", "total_shots_target","B365H", "B365D", "B365A",
        "BWH",   "BWD",   "BWA",
        "IWH",   "IWD",   "IWA",
        "PSH",   "PSD",   "PSA",
        "WHH",   "WHD",   "WHA",
        "VCH",   "VCD",   "VCA",
        # Over/Under 2.5 gols
        "B365>2.5", "B365<2.5",
        "P>2.5",    "P<2.5",
        "Max>2.5",  "Max<2.5",
        "Avg>2.5",  "Avg<2.5",
        # Asian Handicap
        "AHh", "B365AHH", "B365AHA",
        "PAHH", "PAHA",
        "MaxAHH", "MaxAHA",
        "AvgAHH", "AvgAHA",
        
    ], errors="ignore")

    y = df[target]

    # manter apenas números
    X = X.select_dtypes(include=[np.number])
    
    feature_names = X.columns.tolist()
    odds_cols = [
    # Resultado (1X2)
    "B365H", "B365D", "B365A",
    "BWH",   "BWD",   "BWA",
    "IWH",   "IWD",   "IWA",
    "PSH",   "PSD",   "PSA",
    "WHH",   "WHD",   "WHA",
    "VCH",   "VCD",   "VCA",
    # Over/Under 2.5 gols
    "B365>2.5", "B365<2.5",
    "P>2.5",    "P<2.5",
    "Max>2.5",  "Max<2.5",
    "Avg>2.5",  "Avg<2.5",
    # Asian Handicap
    "AHh", "B365AHH", "B365AHA",
    "PAHH", "PAHA",
    "MaxAHH", "MaxAHA",
    "AvgAHH", "AvgAHA",
]
    
    split_date = "2022-07-01"
    
    train_mask = df["Date"] < split_date
    test_mask  = df["Date"] >= split_date
    
    # split
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # preencher NaN usando média do treino
    
    imputer = SimpleImputer(strategy="mean")
    
    X_train = imputer.fit_transform(X_train)  # aprende a média SÓ no treino
    X_test  = imputer.transform(X_test)       # aplica a média do treino no teste
    
    feature_names = [f for f, s in zip(feature_names, imputer.statistics_) if not np.isnan(s)]

    
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    pred = model.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, pred)

    print("AUC:", auc)


        # ---- Inspecionar coeficientes ----
    
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": model.coef_[0],
        "abs_coef": np.abs(model.coef_[0])
    }).sort_values("abs_coef", ascending=False)

    print("\n🔍 Top 30 features por peso absoluto:")
    print(coef_df.head(30).to_string(index=False))

    pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)
    print("\nAUC:", auc)
    
    # targets
    y_train_home = df.loc[train_mask, "HC"]
    y_train_away = df.loc[train_mask, "AC"]
    
    y_test_home = df.loc[test_mask, "HC"]
    y_test_away = df.loc[test_mask, "AC"]
    
    # -------- modelo home corners --------
    
    model_home = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    
    model_home.fit(X_train, y_train_home)
    
    pred_home = model_home.predict(X_test)
    
    mae_home = mean_absolute_error(y_test_home, pred_home)
    rmse_home = np.sqrt(mean_squared_error(y_test_home, pred_home))
    
    print("\n📊 Home Corners")
    print("MAE:", mae_home)
    print("RMSE:", rmse_home)


    # -------- modelo away corners --------
    
    model_away = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    
    model_away.fit(X_train, y_train_away)
    
    pred_away = model_away.predict(X_test)
    
    mae_away = mean_absolute_error(y_test_away, pred_away)
    rmse_away = np.sqrt(mean_squared_error(y_test_away, pred_home))
    
    print("\n📊 Away Corners")
    print("MAE:", mae_away)
    print("RMSE:", rmse_away)
    
    
    # -------- total corners --------
    
    pred_total = pred_home + pred_away
    real_total = y_test_home + y_test_away
    
    mae_total = mean_absolute_error(real_total, pred_total)
    rmse_total = np.sqrt(mean_squared_error(real_total, pred_total))
 
    
    print("\n📊 Total Corners")
    print("MAE:", mae_total)
    print("RMSE:", rmse_total)


##########         CONTROLE         ############
df_base=leitura_e_saneamento()

df_model = build_model_dataset(df_base, windows=[3,5,10])

separacao(df_model)

