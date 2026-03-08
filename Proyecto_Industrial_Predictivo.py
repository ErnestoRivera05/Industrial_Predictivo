import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def url(url):
    return pd.read_csv(url)


staff_line = url(
    "C:/Users/ernes/OneDrive/Documentos/progra/Documents/bloque_5_pandas/Industrial_Predictivo/personal_lineas.csv"
)
historical_production = url(
    "C:/Users/ernes/OneDrive/Documentos/progra/Documents/bloque_5_pandas/Industrial_Predictivo/produccion_historica.csv"
)
historical_sales = url(
    "C:/Users/ernes/OneDrive/Documentos/progra/Documents/bloque_5_pandas/Industrial_Predictivo/ventas_historicas.csv"
)

staff_line
historical_production
historical_sales

# 1) Limpieza & Consistencia


# Revisá nulos/duplicados/tipos.
def nulls(df):
    return df.isnull().sum()


nulls_staff = nulls(staff_line)
nulls_staff
nulls_production = nulls(historical_production)
nulls_production
nulls_sales = nulls(historical_sales)
nulls_sales


def duplicates(df):
    return df.duplicated().sum()


duplicates_staff = duplicates(staff_line)
duplicates_staff
duplicates_production = duplicates(historical_production)
duplicates_production
duplicates_sales = duplicates(historical_sales)
duplicates_sales


def types(df):
    return df.info()


type_staff = types(staff_line)
type_staff
type_production = types(historical_production)
type_production
type_sales = types(historical_sales)
type_sales


# Confirmá consistencia de linea y producto entre archivos (como ya lo hacías con set()).
def production_line(df, name_column):
    return set(df[name_column].unique())


consistency_production = production_line(historical_production, "planta")
consistency_line = production_line(staff_line, "planta")

print(
    "Productos en produccion pero que no están en ventas",
    consistency_production - consistency_line,
)
print(
    "Productos en ventas pero no en produccion",
    consistency_line - consistency_production,
)


# Parseá fecha con pd.to_datetime y ordená por fecha.
def parsea_fecha(df, name_column):
    df[name_column] = pd.to_datetime(df[name_column], format="%Y-%m-%d")
    return df


# ------------- Parsed Production --------------------
parsed_production = parsea_fecha(historical_production, "fecha")
group_production = historical_production.sort_values("fecha", ascending=True)
historical_production
# ---------------- Parsed Sales ------------------------
parsed_sales = parsea_fecha(historical_sales, "fecha")
group_sales = historical_sales.sort_values("fecha", ascending=True)
historical_sales

# 2) Integración (tabla maestra mensual)
# Uní producción + ventas por ["fecha","linea"] (inner).
join_production_sales = pd.merge(
    historical_production, historical_sales, on="producto", how="inner"
)
join_production_sales
# Uní el resultado con personal_lineas por ["linea"] (left).

join_personal_line = pd.merge(
    join_production_sales, staff_line, left_on="linea_x", right_on="linea", how="left"
)

[join_personal_line]


# Validá que no se dupliquen filas y que quede una fila por linea-fecha.
def homogeneizar(df, cambios):
    return df.rename(columns=cambios)


join_personal_line = homogeneizar(
    join_personal_line,
    {"planta_x": "planta", "linea_x": "linea", "fecha_x": "fecha", "linea": "linea_y"},
)
join_personal_line

join_production_sales = homogeneizar(
    join_production_sales,
    {"planta_x": "planta", "linea_x": "linea", "fecha_x": "fecha", "linea": "linea_y"},
)
join_production_sales
# --------- Eliminar linea -------------
# join_personal_line = homogeneizar(join_personal_line, {"linea": "linea_y"})
# join_personal_line


def drop_homogeneizar(df, cambios):
    return df.drop(columns=cambios)


join_personal_line = drop_homogeneizar(
    join_personal_line, ["fecha_y", "planta_y", "linea_y"]
)
join_personal_line

join_production_sales = drop_homogeneizar(join_production_sales, ["fecha_y", "linea_y"])
join_production_sales
# 3) Feature Engineering (claves para el modelo)

# Crea variables por linea:

# Lags: prod_lag1, prod_lag2, ventas_lag1 (shift por línea).
join_personal_line["prod_lag1"] = join_personal_line.groupby("linea")[
    "unidades_producidas"
].shift(1)
join_personal_line

# prod_lag2
join_personal_line["prod_lag2"] = join_personal_line.groupby("linea")[
    "unidades_producidas"
].shift(2)
join_personal_line
# Ventas_lag1
join_production_sales["ventas_lag1"] = join_production_sales.groupby("linea")[
    "unidades_vendidas"
].shift(1)
join_production_sales


# Ratios: eficiencia_prod = unidades_vendidas / unidades_producidas,
def eficiencia_prod(df, column_1, column_2):
    return df[column_1] / df[column_2]


join_production_sales["eficiencia_producida"] = eficiencia_prod(
    join_production_sales, "unidades_vendidas", "unidades_producidas"
)

join_production_sales
# tasa_defectos = defectos / unidades_producidas,
join_production_sales["tasa_defectos"] = eficiencia_prod(
    join_production_sales, "defectos", "unidades_producidas"
)
join_production_sales
# uso_horas = horas_trabajadas / horas_trabajo_disponibles.
join_personal_line["uso_horas"] = eficiencia_prod(
    join_personal_line, "horas_trabajadas", "horas_trabajo_disponibles"
)
join_personal_line


# Calendario: mes, Q (trimestre), is_peak (1 si mes en {6,7,11,12}).


def extract_calendar_features(df, col_fecha):

    df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
    df["mes"] = df[col_fecha].dt.month
    df["Q"] = df[col_fecha].dt.quarter
    df["is_peak"] = np.where(df["mes"].isin([6, 7, 11, 12]), 1, 0)

    return df


join_personal_line = extract_calendar_features(join_personal_line, "fecha")
join_personal_line
join_personal_line[["fecha", "mes", "Q", "is_peak"]].head()


# Costos/Precios: ingreso = unidades_vendidas * precio_venta, costo = unidades_producidas * costo_unitario, margen = ingreso - costo.


def calculation(df, new_column, column_1, column_2, operation="multiply"):
    df[column_1] = pd.to_numeric(df[column_1], errors="coerce")
    df[column_2] = pd.to_numeric(df[column_2], errors="coerce")

    if operation == "multiply":
        df[new_column] = df[column_1] * df[column_2]
    elif operation == "subtract":
        df[new_column] = df[column_1] - df[column_2]
    else:
        raise ValueError("La operación debe ser 'multiply' o 'subtract'")
    return df


join_production_sales = calculation(
    join_production_sales,
    "Ingreso",
    "unidades_vendidas",
    "precio_venta",
)
join_production_sales = calculation(
    join_production_sales,
    "Costo",
    "unidades_producidas",
    "costo_unitario",
)
join_production_sales = calculation(
    join_production_sales, "Margen", "Ingreso", "Costo", "subtract"
)
join_production_sales
# Objetivo (y): unidades_producidas de t+1
# Construí target_next = unidades_producidas.shift(-1) por línea.
join_production_sales["target_next"] = join_production_sales.groupby("linea")[
    "unidades_producidas"
].shift(-1)


# 4) Split temporal (NO aleatorio)
def features(df, target, *columns):
    X = df[list(columns)]
    y = df[target]
    return X, y


dataset_final = pd.merge(
    join_production_sales,
    join_personal_line[["linea", "mes", "Q", "is_peak", "uso_horas"]],
    on="linea",
    how="left",
)

X, y = features(
    dataset_final,
    "target_next",
    "eficiencia_producida",
    "tasa_defectos",
    "Margen",
    "ventas_lag1",
    "mes",
    "Q",
    "is_peak",
)
dataset_final

# Filtrado por fecha: train = 2024-01 … 2025-06, valid = 2025-07 … 2025-09.
train = dataset_final[dataset_final["fecha"] < "2025-07-01"]
valid = dataset_final[
    (dataset_final["fecha"] >= "2025-07-01") & (dataset_final["fecha"] <= "2025-09-30")
]

X_train, y_train = features(
    train,
    "target_next",
    "eficiencia_producida",
    "tasa_defectos",
    "Margen",
    "ventas_lag1",
    "mes",
    "Q",
    "is_peak",
)

X_valid, y_valid = features(
    valid,
    "target_next",
    "eficiencia_producida",
    "tasa_defectos",
    "Margen",
    "ventas_lag1",
    "mes",
    "Q",
    "is_peak",
)


# (Opcional) TimeSeriesSplit(n_splits=3) para validar robusto.

tscv = TimeSeriesSplit(n_splits=3)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
    print(f"Fold {fold+1}")
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

print(type(tscv))

# Limpiar los Nan para evitar errores en el fit()
X_train = X_train.fillna(0)
X_valid = X_valid.fillna(0)
y_train = y_train.fillna(0)
y_valid = y_valid.fillna(0)

# 5) Modelos (baseline + ML)

# Baseline (naive): predice prod_lag1 (lo del mes pasado).

# Regresión Lineal (sklearn.linear_model.LinearRegression).

# (Opcional) Ridge o RandomForestRegressor para comparar.

# Features sugeridas (X):
# [prod_lag1, prod_lag2, ventas_lag1, eficiencia_prod, tasa_defectos, uso_horas, precio_venta, costo_unitario, mes, is_peak, operarios]

# Métrica: MAE, RMSE, R2. Compará siempre vs baseline.
# Crear el modelo
model = LinearRegression()

# Entrenar
model.fit(X_train, y_train)

# Predecir sobre el conjunto de validación
y_pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

print("📈 Evaluación del modelo:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.2f}")

coef = pd.DataFrame(
    {"Variable": X_train.columns, "Coeficiente": model.coef_}
).sort_values(by="Coeficiente", ascending=False)

coef


# 6) Evaluación y Diagnóstico

# Tabla de métricas por modelo y línea.

# Gráfico real vs predicho para 2025-07..09 (valid).

# Residuales: y_true - y_pred → verificá sesgos por línea/mes.
valid["y_pred"] = y_pred
valid["error"] = valid["target_next"] - valid["y_pred"]

# Métricas por línea
metrics_by_line = valid.groupby("linea").apply(
    lambda grp: pd.Series(
        {
            "MAE": np.mean(np.abs(grp["error"])),
            "RMSE": np.sqrt(np.mean(grp["error"] ** 2)),
            "R2": 1
            - np.sum(grp["error"] ** 2)
            / np.sum((grp["target_next"] - grp["target_next"].mean()) ** 2),
        }
    )
)

metrics_by_line

plt.figure(figsize=(12, 5))
plt.plot(valid["fecha"], valid["target_next"], label="Real", marker="o")
plt.plot(valid["fecha"], valid["y_pred"], label="Predicho", marker="x")
plt.title("Real vs Predicción (Validación 2025-07..09)")
plt.xlabel("Fecha")
plt.ylabel("Unidades Producidas")
plt.legend()
plt.grid(True)
plt.show()

valid["residual"] = valid["target_next"] - valid["y_pred"]
valid[["fecha", "linea", "target_next", "y_pred", "residual"]].head()

plt.figure(figsize=(10, 5))
plt.axhline(0, color="black", linestyle="--")
plt.scatter(valid["fecha"], valid["residual"])
plt.title("Residuales en Validación")
plt.ylabel("Error (y_true - y_pred)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

residuals_by_line = valid.groupby("linea")["residual"].agg(
    ["mean", "std", "min", "max"]
)
residuals_by_line

# 7) Forecast 2025-10

# Usá el mejor modelo por línea para predecir 2025-10.
# Última fila disponible por línea (Septiembre 2025)
# 7) FORECAST OCTUBRE 2025

# 1. Tomar la última fila disponible por línea
last_rows = dataset_final.sort_values("fecha").groupby("linea").tail(1)

# 2. Construir las features de entrada para el modelo
X_oct = last_rows[
    [
        "eficiencia_producida",
        "tasa_defectos",
        "Margen",
        "ventas_lag1",
        "mes",
        "Q",
        "is_peak",
    ]
]

# 3. Predicción para octubre 2025
forecast_oct = model.predict(X_oct)

# 4. Crear tabla resumen por línea
forecast_table = pd.DataFrame(
    {"linea": last_rows["linea"].values, "prediccion_oct_2025": forecast_oct}
)

print("📄 Forecast por línea:")
print(forecast_table)

# 5. Resumen total planta
resumen_planta = forecast_table["prediccion_oct_2025"].sum()

print("\n📌 Producción total estimada para Octubre 2025:", resumen_planta)

# ---------------- Gráfico Real vs Predicho + Forecast -----------------

valid_sorted = valid.sort_values("fecha")
y_pred_valid = model.predict(X_valid)

plt.figure(figsize=(10, 6))

plt.plot(valid_sorted["fecha"], y_valid, label="Real (2025-07 a 2025-09)")

plt.plot(
    valid_sorted["fecha"], y_pred_valid, label="Predicho (Validación)", linestyle="--"
)

plt.scatter(
    pd.to_datetime(["2025-10-01"]),
    [forecast_table["prediccion_oct_2025"].mean()],
    color="red",
    s=100,
    label="Forecast Octubre",
)

plt.title("Producción: Real vs Predicho + Forecast Octubre 2025")
plt.xlabel("Fecha")
plt.ylabel("Unidades producidas")
plt.legend()
plt.grid(True)
plt.show()
