# %% [markdown]
# ## 0) Setup & Conversione csv -> parquet

# %%
import os
from glob import glob
from distfit import distfit
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

print("Librerie importate")

# Impostazioni grafiche uniformi per rendere tutte le figure coerenti e leggibili.

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "grid.color": "#CCCCCC",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

sns.set_theme(
    style="whitegrid",  
    palette="tab10",    
    font_scale=1.1
)

plt.rcParams["lines.linewidth"] = 2

# Rimozione outliers
def remove_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series >= lower) & (series <= upper)]

# Directory dati
DATA_DIR = "data"
PARQUET_DIR = "parquet"
FIGS_DIR = "figs"
OUTPUT_DIR = "output"

# Creazione cartelle 
os.makedirs(PARQUET_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# I dataset originali sono costituiti da molti file CSV di grandi dimensioni.
# Per evitare di saturare la RAM in fase di analisi, ogni CSV viene convertito in formato Parquet, che consente:
#   - lettura colonnare,
#   - compressione più efficiente,
#   - streaming in batch attraverso PyArrow.
# Questo rende il progetto scalabile.

def convert_csv_to_parquet_fast(csv_dir, out_subdir):
    out_dir = os.path.join(PARQUET_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted(glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        raise ValueError(f"Nessun CSV in {csv_dir}")

    for f in csv_files:
        name = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(out_dir, f"{name}.parquet")

        if os.path.exists(out_path):
            print(f"[SKIP] {name}")
            continue

        print(f"[CSV → PARQUET] {name}")
        table = pacsv.read_csv(f)
        pq.write_table(table, out_path)

    print(f"Conversione completata per {csv_dir}\n")

# Conversione dei tre dataset
convert_csv_to_parquet_fast(os.path.join(DATA_DIR, "invocations"), "invocations")
convert_csv_to_parquet_fast(os.path.join(DATA_DIR, "durations"),   "durations")
convert_csv_to_parquet_fast(os.path.join(DATA_DIR, "memory"),      "memory")

print("Tutti i CSV convertiti in Parquet")

# %% [markdown]
# ## 1) Caricamento parquet

# %%
# Creazione di tre dataset Arrow distinti. Arrow non carica i file in memoria ma opera in streaming. 
# Ogni dataset corrisponde a una parte del workload:
#   - invocations: volumi temporali minuto per minuto,
#   - durations: tempi di esecuzione delle funzioni,
#   - memory: profili di memoria allocata dalle app.

INV_PATH = os.path.join(PARQUET_DIR, "invocations")
DUR_PATH = os.path.join(PARQUET_DIR, "durations")
MEM_PATH = os.path.join(PARQUET_DIR, "memory")

ds_invocations = ds.dataset(INV_PATH, format="parquet")
ds_durations   = ds.dataset(DUR_PATH, format="parquet")
ds_memory      = ds.dataset(MEM_PATH, format="parquet")

print("Dataset Arrow pronti")

# %% [markdown]
# ## 2) Esplorazione dei Dataset

# %%
# Funzione di ispezione: permette di verificare la struttura dei dataset
# senza caricarli integralmente.
# Utile per effettuare controlli preliminari sulla correttezza dei file caricati.

def quick_info(dataset, name, sample_rows=5):
    print(f"\n=== {name.upper()} ===")

    # Numero record
    n_rows = dataset.count_rows()
    print(f"Record: {n_rows:,}")

    # Prime 10 colonne
    schema_cols = [f"{f.name}:{f.type}" for f in dataset.schema][:10]
    print(f"Colonne (prime 10 su {len(dataset.schema)} totali):")
    for c in schema_cols:
        print(" -", c)

    # Campione
    sample_table = dataset.head(sample_rows)
    df_sample = sample_table.to_pandas()
    
    print("\nPrime righe campione:")
    display(df_sample)

print("Esplorazione veloce con Arrow + batch streaming")

quick_info(ds_invocations, "Invocations")
quick_info(ds_durations,   "Durations")
quick_info(ds_memory,      "Memory")

# %% [markdown]
# ## 3) Panorama generale
# 

# %%
# Estrae solamente le colonne chiave dei tre dataset (Owner, App, Function)
# per valutare dimensione, copertura e sovrapposizione tra le tre fonti dati.

inv_keys = ds_invocations.to_table(columns=["HashOwner", "HashApp", "HashFunction"])
dur_keys = ds_durations.to_table(columns=["HashOwner", "HashApp", "HashFunction"])
mem_keys = ds_memory.to_table(columns=["HashOwner", "HashApp"])

inv_df = inv_keys.to_pandas()
dur_df = dur_keys.to_pandas()
mem_df = mem_keys.to_pandas()

n_owners = inv_df['HashOwner'].nunique()
n_apps   = inv_df['HashApp'].nunique()
n_funcs  = inv_df['HashFunction'].nunique()

print("Cardinalità:")
print(f"- Owner totali:    {n_owners:,}")
print(f"- App totali:      {n_apps:,}")
print(f"- Funzioni totali: {n_funcs:,}\n")

# Verifica quali funzioni hanno il corrispondente record di durata,
# e quali applicazioni presentano dati relativi alla memoria allocata.
# Nota: i tre dataset non sono completamente allineati.

fun_with_dur = set(inv_df["HashFunction"]) & set(dur_df["HashFunction"])
n_fun_dur   = len(fun_with_dur)
n_fun_no_dur = n_funcs - n_fun_dur

app_with_mem = set(inv_df["HashApp"]) & set(mem_df["HashApp"])
n_app_mem   = len(app_with_mem)
n_app_no_mem = n_apps - n_app_mem

print("Copertura dataset:")
print(f"- Funzioni con durations:  {n_fun_dur:,}")
print(f"- Funzioni senza durations: {n_fun_no_dur:,}")
print(f"- App con memory:          {n_app_mem:,}")
print(f"- App senza memory:        {n_app_no_mem:,}\n")

# Creazione CSV riepilogativo

summary_df = pd.DataFrame({
    "Metric": [
        "Owner totali", "App totali", "App con memory", "App senza memory",
        "Funzioni totali", "Funzioni con durations", "Funzioni senza durations"
    ],
    "Value": [
        n_owners, n_apps, n_app_mem, n_app_no_mem,
        n_funcs, n_fun_dur, n_fun_no_dur
    ]
})

summary_df.to_csv(os.path.join(OUTPUT_DIR, "panorama_generale_summary.csv"),
                  index=False, sep=';', encoding='utf-8')

# Costruzione di un riepilogo compatto delle principali dimensioni del dataset.

ordered_metrics = [
    "Owner totali",
    "App totali", "App con memory", "App senza memory",
    "Funzioni totali", "Funzioni con durations", "Funzioni senza durations"
]

summary_df["Metric"] = pd.Categorical(summary_df["Metric"],
                                      categories=ordered_metrics,
                                      ordered=True)
summary_df = summary_df.sort_values("Metric")

category_map = {
    "Owner totali": "Owners",
    "App totali": "Apps",
    "App con memory": "Apps",
    "App senza memory": "Apps",
    "Funzioni totali": "Funzioni",
    "Funzioni con durations": "Funzioni",
    "Funzioni senza durations": "Funzioni"
}

summary_df["Categoria"] = summary_df["Metric"].map(category_map)

category_palette = {
    "Owners":   "#75a4cf",  
    "Apps":     "#85e173", 
    "Funzioni": "#eaa968",  
}

plt.figure(figsize=(11, 6))
ax = sns.barplot(
    x="Value",
    y="Metric",
    data=summary_df,
    hue="Categoria",      
    dodge=False,
    palette=category_palette
)

plt.title("Panorama generale - Cardinalità e copertura dataset")
plt.xlabel("Conteggio")
plt.ylabel("")
plt.legend(title="Categorie", loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "panorama_generale_summary.png"))
plt.show()


# %% [markdown]
# ## 4) Analisi delle Invocazioni

# %%
print("Analisi invocazioni in streaming per RAM limitata...")

# Ogni file delle invocazioni contiene 1440 colonne numeriche, una per ogni minuto della giornata. 
# Per ciascun batch viene calcolato il numero totale di invocazioni
# per funzione e per applicazione sommando le 1440 colonne.
# L'elaborazione è interamente in streaming per evitare l'allocazione di tabelle enormi.

time_cols = [str(i) for i in range(1, 1441)]
key_cols = ["HashOwner", "HashApp", "HashFunction"]
all_cols = key_cols + time_cols

func_counter = Counter()
app_counter  = Counter()
time_series_total = np.zeros(1440, dtype=np.int64)

# Analisi in streaming: ogni batch contiene solo una frazione dei dati,
# permettendo di elaborare l’intero dataset senza mai caricare tutto in RAM.

for batch in ds_invocations.to_batches(columns=all_cols):
    batch_df = batch.to_pandas()
    
    # Totale invocazioni per funzione
    func_counter.update(batch_df.groupby("HashFunction")[time_cols].sum().sum(axis=1).to_dict())
    
    # Totale invocazioni per app
    app_counter.update(batch_df.groupby("HashApp")[time_cols].sum().sum(axis=1).to_dict())
    
    # Serie temporale totale
    time_series_total += batch_df[time_cols].sum(axis=0).to_numpy()


# Top 10 funzioni più invocate
func_counts = pd.Series(func_counter).sort_values(ascending=False)
top10_funcs = pd.DataFrame({"Function": func_counts.index[:10], "Count": func_counts.values[:10]})
top10_funcs.to_csv(os.path.join(OUTPUT_DIR, "top10_functions.csv"), index=False, sep=';', encoding='utf-8')

plt.figure(figsize=(10,6))
ax = sns.barplot(x="Count", y="Function", data=top10_funcs, hue="Function", dodge=False, legend=False)
plt.title("Top 10 funzioni più invocate")
plt.xlabel("Numero invocazioni")
plt.ylabel("HashFunction")
plt.legend([], [], frameon=False)

for i, row in top10_funcs.iterrows():
    ax.text(row['Count'] + top10_funcs['Count'].max()*0.01, i, f"{row['Count']:,}", va='center')

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "top10_functions.png"))
plt.show()

# Top 3 funzioni più invocate
print("\nTop 3 funzioni più invocate:")
for i, row in top10_funcs.head(3).iterrows():
    print(f"{i+1}. {row['Function']} - {row['Count']:,} invocazioni")

# Top 10 app più invocate
app_counts = pd.Series(app_counter).sort_values(ascending=False)
top10_apps = pd.DataFrame({"App": app_counts.index[:10], "Count": app_counts.values[:10]})
top10_apps.to_csv(os.path.join(OUTPUT_DIR, "top10_apps.csv"), index=False, sep=';', encoding='utf-8')

plt.figure(figsize=(10,6))
ax = sns.barplot(x="Count", y="App", data=top10_apps, hue="App", dodge=False, legend=False)
plt.title("Top 10 App più invocate")
plt.xlabel("Numero invocazioni")
plt.ylabel("HashApp")
plt.legend([], [], frameon=False)

for i, row in top10_apps.iterrows():
    ax.text(row['Count'] + top10_apps['Count'].max()*0.01, i, f"{row['Count']:,}", va='center')

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "top10_apps.png"))
plt.show()

# Top 3 app più invocate
print("\nTop 3 app più invocate:")
for i, row in top10_apps.head(3).iterrows():
    print(f"{i+1}. {row['App']} - {row['Count']:,} invocazioni")

# Aggregazione dei valori dei 1440 minuti su tutte le funzioni per produrre
# la serie temporale delle invocazioni nell’arco della giornata.
minutes = np.arange(1440)
hours = minutes / 60

plt.figure(figsize=(14,5))
plt.plot(hours, time_series_total, label="Invocazioni per minuto", color="blue")
plt.title("Invocazioni totali per minuto della giornata")
plt.xlabel("Ora della giornata")
plt.ylabel("Numero invocazioni")
plt.xticks(np.arange(0,25,1))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "total_time_series.png"))
plt.show()

# Invocazioni aggregate per ora
hourly_series = time_series_total.reshape(24, 60).sum(axis=1)
hourly_df = pd.DataFrame({"Hour": np.arange(24), "Count": hourly_series})
hourly_df.to_csv(os.path.join(OUTPUT_DIR, "hourly_total.csv"), index=False, sep=';', encoding='utf-8')

plt.figure(figsize=(10,5))
sns.barplot(x="Hour", y="Count", data=hourly_df, hue="Hour", dodge=False, legend=False)
plt.title("Invocazioni totali per ora")
plt.xlabel("Ora della giornata")
plt.ylabel("Numero invocazioni")
plt.legend([], [], frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "hourly_total.png"))
plt.show()

# Serie oraria normalizzata sulla media
hourly_norm = hourly_series / hourly_series.mean()

norm_df = pd.DataFrame({
    "Hour": np.arange(24),
    "Normalized": hourly_norm
})

plt.figure(figsize=(10,5))
sns.lineplot(x="Hour", y="Normalized", data=norm_df, marker="o")
plt.title("Invocazioni per ora normalizzate sulla media")
plt.xlabel("Ora della giornata")
plt.ylabel("Invocazioni normalizzate (÷ media)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "hourly_normalized.png"))
plt.show()

# Picco/minimo
peak_minute = np.argmax(time_series_total)
min_minute = np.argmin(time_series_total)
print(f"\nMinuto di picco: {peak_minute} ({peak_minute//60}:{peak_minute%60})")
print(f"Minuto minimo: {min_minute} ({min_minute//60}:{min_minute%60})")

print("Analisi delle invocazioni completata. Figure e CSV salvati in 'figs/' e 'output/'")

# %% [markdown]
# ## 4b) Analisi dei trigger

# %%
# In questa sezione si analizza:
# - Quante funzioni appartengono a ciascun trigger (grafico descrittivo)
# - La distribuzione globale del numero di invocazioni per funzione (PDF)
# - La CDF globale
# - Le CDF separate per ciascuno dei 7 trigger
#
# NOTA:
# Si riutilizza il calcolo delle invocazioni totali ottenuto in Sezione 4:

print("Analisi trigger in corso...")

# Tabella Funzione e Totale invocazioni

func_inv_df = pd.DataFrame({
    "HashFunction": list(func_counter.keys()),
    "TotalInvocations": list(func_counter.values())
})

print(f"Funzioni totali con invocazioni: {len(func_inv_df):,}")

# Aggiunta Trigger a ciascuna funzione

func_trigger_map = (
    ds_invocations.to_table(columns=["HashFunction", "Trigger"])
    .to_pandas()
    .drop_duplicates()
)

func_inv_df = func_inv_df.merge(func_trigger_map, on="HashFunction", how="left")
func_inv_df = func_inv_df.dropna(subset=["Trigger"])

print("Esempio righe unite:")
display(func_inv_df.head())

# Distribuzione globale
all_invocation_totals = func_inv_df["TotalInvocations"].values

# Distribuzione per trigger
invocations_by_trigger = {
    trig: grp["TotalInvocations"].values
    for trig, grp in func_inv_df.groupby("Trigger")
}

# Numero di funzioni per trigger
trigger_func_count = (
    func_inv_df.groupby("Trigger")["HashFunction"]
    .count()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12,6))
sns.barplot(
    x=trigger_func_count.values,
    y=trigger_func_count.index,
    orient="h",
    palette="viridis"
)
plt.title("Numero di funzioni per trigger")
plt.xlabel("Numero di funzioni")
plt.ylabel("Trigger")
plt.grid(axis='x', alpha=0.3, ls='--')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "trigger_num_functions.png"))
plt.show()

# CDF globale

plt.figure(figsize=(10,6))
sns.ecdfplot(all_invocation_totals)
plt.xscale("log")
plt.title("CDF globale delle invocazioni per funzione")
plt.xlabel("Invocazioni totali per funzione (scala log)")
plt.ylabel("CDF")
plt.grid(True, ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "trigger_cdf_global.png"))
plt.show()

# CDF per trigger (comparativa)

plt.figure(figsize=(12,7))

palette = sns.color_palette("tab10", n_colors=len(invocations_by_trigger))

for (trig, vals), col in zip(invocations_by_trigger.items(), palette):

    if len(vals) > 2000:
        sample = np.quantile(vals, np.linspace(0, 1, 2000))
    else:
        sample = np.sort(vals)

    sns.ecdfplot(
        sample,
        label=trig,
        linewidth=2.3,
        alpha=0.85,
        color=col
    )

plt.xscale("log")
plt.ylim(0,1)
plt.title("CDF delle invocazioni per trigger")
plt.xlabel("Invocazioni totali per funzione (scala log)")
plt.ylabel("CDF")

plt.grid(True, ls="--", alpha=0.25)

plt.legend(
    title="Trigger",
    ncol=2,
    fontsize=10,
    title_fontsize=11,
    loc="lower right"
)

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "trigger_cdf_by_type_readable.png"))
plt.show()

plt.figure(figsize=(14,10))

for i, (trig, vals) in enumerate(invocations_by_trigger.items(), 1):
    plt.subplot(3, 3, i)
    sns.ecdfplot(vals)
    plt.xscale("log")
    plt.title(trig)
    plt.xlabel("")
    plt.ylabel("CDF")
    plt.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "trigger_cdf_subplots.png"))
plt.show()

print("Analisi trigger completata.")

# %% [markdown]
# ## 5) Analisi delle Durate

# %%
# In questa sezione viene analizzata la distribuzione delle durate di esecuzione
# delle funzioni serverless.
# L’obiettivo è:
#  - caratterizzare il comportamento globale delle durate,
#  - identificare funzioni e applicazioni con tempi di esecuzione elevati,
#  - studiare la forma della distribuzione (code pesanti, asimmetria),
#  - stimare una distribuzione teorica che approssimi i dati osservati.

print("Analisi delle durate in streaming per RAM limitata... ")

# Contatori e raccolta durate per funzione/app
func_durations = defaultdict(list)
app_durations  = defaultdict(list)
global_durations = []

for batch in ds_durations.to_batches(columns=key_cols + ["Average"]):
    df = batch.to_pandas()
    df = df[df["Average"] >= 0]

    global_durations.extend(df["Average"].tolist())

    for func, group in df.groupby("HashFunction"):
        func_durations[func].extend(group["Average"].tolist())

    for app, group in df.groupby("HashApp"):
        app_durations[app].extend(group["Average"].tolist())


# Statistiche globali
global_series = pd.Series(global_durations)
stats = {
    "Median": global_series.median(),
    "P90": global_series.quantile(0.9),
    "P95": global_series.quantile(0.95),
    "P99": global_series.quantile(0.99),
    "Mean": global_series.mean(),
    "Min": global_series.min(),
    "Max": global_series.max()
}
print("Statistiche globali delle durate (ms):")
for k,v in stats.items():
    print(f"- {k}: {v:.2f}")


# Stima automatica della distribuzione teorica delle durate
# Diverse distribuzioni candidate vengono confrontate tramite
# test di Kolmogorov–Smirnov; la coda estrema è limitata (p99.5)
# per migliorare la stabilità del fit.

print("\nStima automatica della distribuzione delle durate (Average)...")

# si usano solo valori positivi e limitazione della coda estrema

duration_series = pd.Series(global_durations)
duration_series = duration_series[duration_series > 0]

duration_fit = duration_series.clip(upper=duration_series.quantile(0.995))

# inizializzazione distfit
dist = distfit(
    distr=[
        'norm',
        'lognorm',
        'gamma',
        'expon',
        'weibull_min'
    ],
    stats='ks'   # Kolmogorov–Smirnov
)

# stima
dist.fit_transform(duration_fit.values)

# risultati
print("\nDistribuzione migliore stimata:")
print(f"- Nome: {dist.model['name']}")
print(f"- Parametri: {dist.model['params']}")
print(f"- KS statistic: {dist.model['score']:.4f}")

# Salvataggio risultati stima distribuzione teorica
dist_summary = pd.DataFrame([{
    "Metric": "BestFitDistribution",
    "Value": dist.model["name"]
}, {
    "Metric": "Parameters",
    "Value": str(dist.model["params"])
}, {
    "Metric": "KS_statistic",
    "Value": dist.model["score"]
}, {
    "Metric": "NumSamplesUsed",
    "Value": len(duration_fit)
}, {
    "Metric": "ClippingPercentile",
    "Value": 0.995
}, {
    "Metric": "CandidateDistributions",
    "Value": ", ".join(dist.distr)
}])

dist_summary.to_csv(
    os.path.join(OUTPUT_DIR, "duration_distribution_fit_summary.csv"),
    index=False,
    sep=";",
    encoding="utf-8"
)

# plot comparativo (PDF empirica vs distribuzione teorica)
dist.plot()

plt.xscale("log")
plt.xlabel("Durata media (ms, scala log)")
plt.ylabel("Frequenza / densità stimata")
plt.title("Confronto tra PDF empirica e distribuzione teorica stimata")

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "duration_best_fit_distribution.png"))
plt.show()

# Le durate (Average, Minimum, Maximum) sono distribuite su più ordini di grandezza,
# per questo la CDF viene rappresentata in scala logaritmica. 
print("\Generazione CDF per Average, Minimum e Maximum")

avg_list = []
min_list = []
max_list = []

# Lettura in streaming anche dei campi Minimum e Maximum
for batch in ds_durations.to_batches(columns=["Average", "Minimum", "Maximum"]):
    df = batch.to_pandas()
    df = df[(df["Average"] > 0) & (df["Minimum"] > 0) & (df["Maximum"] > 0)]
    
    avg_list.extend(df["Average"].tolist())
    min_list.extend(df["Minimum"].tolist())
    max_list.extend(df["Maximum"].tolist())

plt.figure(figsize=(8,5))

for vals, label in [
    (avg_list, "Average"),
    (min_list, "Minimum"),
    (max_list, "Maximum")
]:
    s = np.sort(vals)
    y = np.linspace(0, 1, len(s))
    plt.plot(s, y, label=label)

plt.xscale("log")
plt.xlabel("Durata (ms, scala log)")
plt.ylabel("CDF")
plt.title("CDF di Average, Minimum e Maximum")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "durations_avg_min_max_cdf.png"))
plt.show()

# Percentili globali 
percentiles = [25, 50, 75, 90, 95, 99]
percentile_values = global_series.quantile([p/100 for p in percentiles])

plt.figure(figsize=(8,4))
sns.barplot(x=[f"P{p}" for p in percentiles], y=percentile_values.values, color="orange")
plt.yscale("log")
plt.title("Percentili globali delle durate medie delle funzioni")
plt.ylabel("Durata media (ms, scala log)")
plt.xlabel("Percentile")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "duration_global_percentiles_log.png"))
plt.show()

# Calcolo Top 10 funzioni/app più lente 

func_median = {f: pd.Series(d).median() for f, d in func_durations.items() if len(d) > 0}
top_slowest_funcs = pd.Series(func_median).sort_values(ascending=False).head(10)

app_median = {a: pd.Series(d).median() for a, d in app_durations.items() if len(d) > 0}
top_slowest_apps = pd.Series(app_median).sort_values(ascending=False).head(10)

# Salvataggio CSV top10
pd.DataFrame({
    "Function": top_slowest_funcs.index,
    "MedianDuration_ms": top_slowest_funcs.values
}).to_csv(os.path.join(OUTPUT_DIR, "top10_slowest_functions.csv"), index=False, sep=";", encoding="utf-8")

pd.DataFrame({
    "App": top_slowest_apps.index,
    "MedianDuration_ms": top_slowest_apps.values
}).to_csv(os.path.join(OUTPUT_DIR, "top10_slowest_apps.csv"), index=False, sep=";", encoding="utf-8")

print("\nTop 3 funzioni più lente:")
for i, (f, v) in enumerate(top_slowest_funcs.head(3).items(), 1):
    print(f"{i}. {f} – {v:.2f} ms")

print("\nTop 3 app più lente:")
for i, (a, v) in enumerate(top_slowest_apps.head(3).items(), 1):
    print(f"{i}. {a} – {v:.2f} ms")

# Boxplot TOP 10 Funzioni più lente – SENZA OUTLIER

rows = []
for f in top_slowest_funcs.index:
    clean_series = remove_outliers_iqr(pd.Series(func_durations[f]))
    for v in clean_series:
        rows.append((f, v))

top_funcs_df = pd.DataFrame(rows, columns=["Function", "Duration"])
top_funcs_df["ShortID"] = top_funcs_df["Function"].str[:10] + "..."

plt.figure(figsize=(16, 9))
sns.boxplot(
    data=top_funcs_df,
    x="ShortID",
    y="Duration",
    color="skyblue",
    linewidth=0.7
)

plt.yscale("log")
plt.title("Distribuzione durate medie – Top 10 funzioni più lente (senza outlier)")
plt.xlabel("HashFunction (abbreviato)")
plt.ylabel("Durata media (ms, scala log)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "top10_slowest_functions_boxplot_clean_log.png"))
plt.show()

# Boxplot TOP 10 Applicazioni più lente – SENZA OUTLIER

rows = []
for a in top_slowest_apps.index:
    clean_series = remove_outliers_iqr(pd.Series(app_durations[a]))
    for v in clean_series:
        rows.append((a, v))

top_apps_df = pd.DataFrame(rows, columns=["App", "Duration"])
top_apps_df["ShortID"] = top_apps_df["App"].str[:10] + "..."

plt.figure(figsize=(16, 10))
sns.boxplot(
    data=top_apps_df,
    x="ShortID",
    y="Duration",
    color="lightgreen",
    linewidth=0.7
)

plt.yscale("log")
plt.title("Distribuzione durate medie – Top 10 applicazioni più lente (senza outlier)")
plt.xlabel("HashApp (abbreviato)")
plt.ylabel("Durata media (ms, scala log)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "top10_slowest_apps_boxplot_clean_log.png"))
plt.show()

print("Analisi delle durate completata. Figure e CSV salvati in 'figs/' e 'output/'")

# %% [markdown]
# ## 6) Analisi delle Memorie

# %%
print("Analisi memoria allocata in streaming per RAM limitata...")

key_cols = ["HashOwner", "HashApp"]
mem_cols = [
    "SampleCount", "AverageAllocatedMb",
    "AverageAllocatedMb_pct5", "AverageAllocatedMb_pct25",
    "AverageAllocatedMb_pct50", "AverageAllocatedMb_pct75",
    "AverageAllocatedMb_pct95", "AverageAllocatedMb_pct99",
    "AverageAllocatedMb_pct100"
]
all_cols = key_cols + mem_cols

# Il dataset della memoria contiene diverse statistiche percentili pre-calcolate.
# Si aggregano i valori per app, in particolar modo su:
#   - mediana,
#   - pct95,
#   - pct99.

app_memory = defaultdict(list)
app_pct95 = defaultdict(list)
app_pct99 = defaultdict(list)
app_samplec = defaultdict(list)
global_memory = []

# Streaming dei batch
for batch in ds_memory.to_batches(columns=all_cols):
    df = batch.to_pandas()
    if "AverageAllocatedMb" not in df.columns:
        continue
    df = df[df["AverageAllocatedMb"].notna() & (df["AverageAllocatedMb"] >= 0)]

    global_memory.extend(df["AverageAllocatedMb"].tolist())

    for app, grp in df.groupby("HashApp"):
        vals = grp["AverageAllocatedMb"].dropna().tolist()
        if vals:
            app_memory[app].extend(vals)
        if "AverageAllocatedMb_pct95" in grp.columns:
            app_pct95[app].extend(grp["AverageAllocatedMb_pct95"].dropna().tolist())
        if "AverageAllocatedMb_pct99" in grp.columns:
            app_pct99[app].extend(grp["AverageAllocatedMb_pct99"].dropna().tolist())
        if "SampleCount" in grp.columns:
            app_samplec[app].extend(grp["SampleCount"].dropna().tolist())

# Statistiche globali memoria
global_series = pd.Series(global_memory)
mem_stats = {
    "Median": global_series.median(),
    "P90": global_series.quantile(0.90),
    "P95": global_series.quantile(0.95),
    "P99": global_series.quantile(0.99),
    "Mean": global_series.mean(),
    "Min": global_series.min(),
    "Max": global_series.max(),
}
print("Statistiche globali della memoria allocata (MB):")
for k, v in mem_stats.items():
    print(f"- {k}: {v:.2f}")

# Percentili globali
percentiles = [25, 50, 75, 90, 95, 99]
percentile_values = global_series.quantile([p / 100 for p in percentiles])

pd.DataFrame({
    "Percentile": [f"P{p}" for p in percentiles],
    "Value_MB": percentile_values.values
}).to_csv(
    os.path.join(OUTPUT_DIR, "memory_global_percentiles.csv"),
    index=False, sep=";", encoding="utf-8"
)

plt.figure(figsize=(8, 4))
sns.barplot(
    x=[f"P{p}" for p in percentiles],
    y=percentile_values.values,
    color="tab:orange"
)
plt.yscale("log")
plt.title("Percentili globali della memoria allocata")
plt.ylabel("Memoria allocata (MB, scala log)")
plt.xlabel("Percentile")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "memory_global_percentiles_log.png"))
plt.show()

# Top 10 app per mediana e pct99

app_median_mem = {
    a: pd.Series(v).median()
    for a, v in app_memory.items()
    if len(v) > 0
}
top10_apps_by_median = (
    pd.Series(app_median_mem).sort_values(ascending=False).head(10)
)

app_pct99_median = {
    a: (pd.Series(v).median() if len(v) > 0 else np.nan)
    for a, v in app_pct99.items()
}
top10_apps_by_pct99 = (
    pd.Series(app_pct99_median).sort_values(ascending=False).head(10)
)

pd.DataFrame({
    "App": top10_apps_by_median.index,
    "MedianMemory_MB": top10_apps_by_median.values
}).to_csv(
    os.path.join(OUTPUT_DIR, "top10_apps_by_median_memory.csv"),
    index=False, sep=";", encoding="utf-8"
)

pd.DataFrame({
    "App": top10_apps_by_pct99.index,
    "Pct99Memory_MB": top10_apps_by_pct99.values
}).to_csv(
    os.path.join(OUTPUT_DIR, "top10_apps_by_pct99_memory.csv"),
    index=False, sep=";", encoding="utf-8"
)

# I boxplot sono costruiti dopo la rimozione degli outlier tramite metodo IQR
# Boxplot mediana
rows = []
for app in top10_apps_by_median.index:
    clean = remove_outliers_iqr(pd.Series(app_memory[app]))
    for v in clean:
        rows.append((app, v))

top10_median_df = pd.DataFrame(rows, columns=["App", "Memory_MB"])
top10_median_df["ShortID"] = top10_median_df["App"].str[:10] + "..."

plt.figure(figsize=(16, 10))
sns.boxplot(
    data=top10_median_df,
    x="ShortID",
    y="Memory_MB",
    color="salmon",
    linewidth=0.7
)
plt.yscale("log")
plt.title("Distribuzione della memoria allocata – Top 10 app per mediana")
plt.xlabel("Applicazione (Hash abbreviato)")
plt.ylabel("Memoria allocata (MB, scala log)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "top10_apps_median_memory_boxplot_clean_log.png"))
plt.show()

# Boxplot pct99
rows = []
for app in top10_apps_by_pct99.index:
    clean = remove_outliers_iqr(pd.Series(app_pct99[app]))
    for v in clean:
        rows.append((app, v))

top10_pct99_df = pd.DataFrame(rows, columns=["App", "Pct99_MB"])
top10_pct99_df["ShortID"] = top10_pct99_df["App"].str[:10] + "..."

plt.figure(figsize=(16, 10))
sns.boxplot(
    data=top10_pct99_df,
    x="ShortID",
    y="Pct99_MB",
    color="skyblue",
    linewidth=0.7
)
plt.yscale("log")
plt.title("Distribuzione della memoria (percentile 99) – Top 10 app")
plt.xlabel("Applicazione (Hash abbreviato)")
plt.ylabel("Memoria Pct99 (MB, scala log)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "top10_apps_pct99_memory_boxplot_clean_log.png"))
plt.show()

# Istogramma globale
plt.figure(figsize=(10, 4))
plt.hist(global_series, bins=80)
plt.title("Istogramma globale della memoria allocata")
plt.xlabel("Memoria allocata (MB)")
plt.ylabel("Conteggio")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "memory_global_hist.png"))
plt.show()

# ECDF globale
sorted_vals = np.sort(global_series.dropna())
ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

plt.figure(figsize=(8, 4))
plt.plot(sorted_vals, ecdf)
plt.title("ECDF della memoria allocata")
plt.xlabel("Memoria allocata (MB)")
plt.ylabel("ECDF")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "memory_global_ecdf.png"))
plt.show()

print("Analisi memoria completata. Figure e CSV salvati in 'figs/' e 'output/'")


# %% [markdown]
# ## 7) Analisi Concatenazione

# %%
# Questa sezione analizza la struttura interna delle applicazioni serverless,
# studiando come le funzioni sono concatenate all’interno di ciascuna app.
# In particolare si osservano:
#  - il numero di funzioni per applicazione,
#  - il numero e la combinazione dei trigger utilizzati,
#  - la presenza di trigger predominanti in termini di carico.

print("Analisi estesa della concatenazione delle funzioni...")

# Estratto delle funzioni: HashApp – HashFunction – Trigger

inv_simple = (
    ds_invocations
    .to_table(columns=["HashApp", "HashFunction", "Trigger"])
    .to_pandas()
    .drop_duplicates()
)

# Numero di funzioni per applicazione

func_per_app = inv_simple.groupby("HashApp")["HashFunction"].nunique()

plt.figure(figsize=(12,6))
sns.histplot(func_per_app.clip(upper=30), bins=30, color="skyblue")
plt.yscale("log")
plt.title("Distribuzione del numero di funzioni per applicazione (clipped a 30)")
plt.xlabel("Numero di funzioni per applicazione")
plt.ylabel("Numero di applicazioni (log)")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "concat_num_functions_per_app.png"))
plt.show()

print("Statistiche funzioni per app:")
print(func_per_app.describe())

# Numero di trigger distinti per applicazione

triggers_per_app = inv_simple.groupby("HashApp")["Trigger"].unique()
num_trig_per_app = triggers_per_app.apply(len)

plt.figure(figsize=(10,5))
sns.countplot(x=num_trig_per_app, color="lightgreen")
plt.title("Numero di trigger distinti per applicazione")
plt.xlabel("Trigger distinti")
plt.ylabel("Numero applicazioni")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "concat_num_triggers_per_app.png"))
plt.show()

# Pattern dei trigger – Top combinazioni

trigger_patterns = pd.Series([tuple(sorted(v)) for v in triggers_per_app.values])
pattern_counts = trigger_patterns.value_counts()

top20_patterns = pattern_counts.head(20)

plt.figure(figsize=(14,7))
sns.barplot(
    y=top20_patterns.index.astype(str),
    x=top20_patterns.values,
    color="lightcoral"
)
plt.title("Top 20 pattern di combinazioni di trigger nelle applicazioni")
plt.xlabel("Numero di applicazioni")
plt.ylabel("Pattern trigger")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "concat_trigger_patterns.png"))
plt.show()

# Correlazione: numero di funzioni vs numero di trigger

plt.figure(figsize=(8,6))
plt.scatter(num_trig_per_app, func_per_app, alpha=0.4)
plt.title("Numero di trigger distinti vs numero di funzioni per app")
plt.xlabel("Trigger distinti")
plt.ylabel("Numero funzioni")
plt.grid(True, ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "concat_trig_vs_funcs.png"))
plt.show()


# Funzioni più invocate per ogni applicazione

# utilizziamo func_counter calcolato nella sezione 4
app_fcalls = {}

func_to_app = {
    r.HashFunction: r.HashApp
    for r in inv_simple.itertuples()
}

for func, total in func_counter.items():
    if func in func_to_app:
        app = func_to_app[func]
        app_fcalls.setdefault(app, []).append((func, total))

# funzione più invocata per app
most_invoked_rows = []

for app, lst in app_fcalls.items():
    func, tot = max(lst, key=lambda x: x[1])
    most_invoked_rows.append({
        "HashApp": app,
        "HashFunction": func,
        "TotalInvocations": tot
    })

most_invoked_df = pd.DataFrame(most_invoked_rows)

plt.figure(figsize=(10,6))
sns.histplot(most_invoked_df["TotalInvocations"], bins=80, log_scale=True, color="orange")
plt.title("Distribuzione della funzione più invocata per ciascuna applicazione")
plt.xlabel("Invocazioni (scala log)")
plt.ylabel("Numero applicazioni")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "concat_most_invoked_per_app.png"))
plt.show()

# Classificazione applicazioni per trigger predominante

# per ogni app: somma invocazioni delle sue funzioni per trigger
app_trigger_load = []

# serve una mappatura HashFunction -> Trigger
func_trigger_map = {
    r.HashFunction: r.Trigger
    for r in inv_simple.itertuples()
}

for app, funclist in app_fcalls.items():
    trig_load = {}
    for func, tot in funclist:
        trig = func_trigger_map.get(func)
        trig_load[trig] = trig_load.get(trig, 0) + tot

    if trig_load:
        predominant_trigger = max(trig_load.items(), key=lambda x: x[1])[0]
        app_trigger_load.append((app, predominant_trigger))

app_trigger_df = pd.DataFrame(app_trigger_load, columns=["HashApp", "PredominantTrigger"])

plt.figure(figsize=(10,5))
sns.countplot(
    y=app_trigger_df["PredominantTrigger"],
    order=app_trigger_df["PredominantTrigger"].value_counts().index,
    color="mediumpurple"
)
plt.title("Trigger predominante per applicazione")
plt.xlabel("Numero applicazioni")
plt.ylabel("Trigger")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "concat_predominant_trigger.png"))
plt.show()

print("Analisi estesa della concatenazione completata.")


# %% [markdown]
# ## 8) Analisi incrociata Invocations–Durations–Memory

# %%
print("\nAnalisi Incrociata (invocazioni ⨝ durate ⨝ memoria)")

# I tre dataset non possono essere uniti direttamente a causa delle elevate dimensioni complessive.
# Per questo motivo viene costruito manualmente un campione incrociato nel seguente modo:
#   1. estrazione delle chiavi comuni tra invocations, durations e memory;
#   2. costruzione di due dizionari parziali (durate e memoria) in streaming;
#   3. scansione delle invocations in streaming per generare un campione coerente.

# Ogni record del campione contiene:
#   - numero totale di invocazioni,
#   - durata media della funzione,
#   - memoria media allocata dall’applicazione,
#   - numero di campioni di durata raccolti.

# Questo campione permette di esplorare correlazioni e pattern tra
# intensità di utilizzo, prestazioni e footprint di memoria.
# L’obiettivo non è ricostruire l’intero join tra dataset (computazionalmente troppo costoso),
# ma ottenere un sottoinsieme statisticamente rappresentativo del workload.

inv_keys = set()    
dur_keys = set()
mem_keys = set()

print("\nCaricamento chiavi invocations...")
for batch in ds_invocations.to_batches(columns=["HashOwner","HashApp","HashFunction"]):
    df = batch.to_pandas()
    inv_keys.update(zip(df.HashOwner, df.HashApp, df.HashFunction))
print(f"   → funzioni in invocations: {len(inv_keys):,}")

print("\nCaricamento chiavi durations...")
for batch in ds_durations.to_batches(columns=["HashOwner","HashApp","HashFunction"]):
    df = batch.to_pandas()
    dur_keys.update(zip(df.HashOwner, df.HashApp, df.HashFunction))
print(f"   → funzioni in durations: {len(dur_keys):,}")

print("\nCaricamento chiavi memoria...")
for batch in ds_memory.to_batches(columns=["HashOwner","HashApp"]):
    df = batch.to_pandas()
    mem_keys.update(zip(df.HashOwner, df.HashApp))
print(f"   → applicazioni in memory: {len(mem_keys):,}")

# Coverage e intersezioni
common_inv_dur = inv_keys & dur_keys
common_all = {(o,a,f) for (o,a,f) in common_inv_dur if (o,a) in mem_keys}

print("\nCopertura tra dataset (chiavi accoppiabili)")
print(f" - Invocations ∩ Durations: {len(common_inv_dur):,}")
print(f" - Invocations ∩ Durations ∩ Memory: {len(common_all):,}")

# Costruzione campione incrociato
target_size = 30_000
sample = []

# Dizionari parziali
dur_dict = {}
for batch in ds_durations.to_batches(columns=["HashOwner","HashApp","HashFunction","Average","Count"]):
    df = batch.to_pandas()
    for r in df.itertuples(index=False):
        dur_dict[(r.HashOwner, r.HashApp, r.HashFunction)] = (r.Average, r.Count)
    if len(dur_dict) > 100_000:
        break

mem_dict = {}
for batch in ds_memory.to_batches(columns=["HashOwner","HashApp","AverageAllocatedMb"]):
    df = batch.to_pandas()
    for r in df.itertuples(index=False):
        mem_dict[(r.HashOwner, r.HashApp)] = r.AverageAllocatedMb
    if len(mem_dict) > 50_000:
        break

# Sampling delle invocations
for batch in ds_invocations.to_batches():
    df = batch.to_pandas()
    minute_cols = [c for c in df.columns if c.isdigit()]
    df["TotalInvocations"] = df[minute_cols].sum(axis=1)

    for r in df.itertuples(index=False):
        key = (r.HashOwner, r.HashApp, r.HashFunction)
        if key not in dur_dict:
            continue
        if (r.HashOwner, r.HashApp) not in mem_dict:
            continue
        sample.append((
            r.TotalInvocations,
            dur_dict[key][0],
            dur_dict[key][1],
            mem_dict[(r.HashOwner, r.HashApp)]
        ))
    if len(sample) >= target_size:
        break

sample_df = pd.DataFrame(sample, columns=[
    "Invocations",
    "AvgDuration_ms",
    "CountDurSamples",
    "AvgMem_MB"
])
print(f"   → campione creato: {len(sample_df):,} righe")

# Statistiche incrociate
print("\nSTATISTICHE INCROCIATE")
print(sample_df.describe(percentiles=[0.5, 0.9, 0.99]))

# Scatter lineari
plt.figure(figsize=(7,5))
plt.scatter(sample_df["Invocations"], sample_df["AvgDuration_ms"], s=8, alpha=0.35)
plt.title("Invocazioni vs durata media")
plt.xlabel("Numero di invocazioni")
plt.ylabel("Durata media (ms)")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "inv_vs_duration.png"))
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(sample_df["Invocations"], sample_df["AvgMem_MB"], s=8, alpha=0.35)
plt.title("Invocazioni vs memoria media allocata")
plt.xlabel("Numero di invocazioni")
plt.ylabel("Memoria media (MB)")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "inv_vs_memory.png"))
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(sample_df["AvgDuration_ms"], sample_df["AvgMem_MB"], s=8, alpha=0.35)
plt.title("Durata media vs memoria media")
plt.xlabel("Durata media (ms)")
plt.ylabel("Memoria media (MB)")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "duration_vs_memory.png"))
plt.show()

# Scatter log-log
sample_df_filtered = sample_df[(sample_df > 0).all(axis=1)]

plt.figure(figsize=(7,5))
plt.scatter(sample_df_filtered["Invocations"], sample_df_filtered["AvgDuration_ms"], s=8, alpha=0.35)
plt.xscale("log")
plt.yscale("log")
plt.title("Invocazioni vs durata media (scala log-log)")
plt.xlabel("Invocazioni (scala log)")
plt.ylabel("Durata media (ms, scala log)")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "inv_vs_duration_loglog.png"))
plt.show()

# CDF metriche principali
plt.figure(figsize=(8,5))
for col, label in [
    ("Invocations", "Invocazioni"),
    ("AvgDuration_ms", "Durata media (ms)"),
    ("AvgMem_MB", "Memoria media (MB)")
]:
    vals = sample_df_filtered[col].sort_values()
    y = np.linspace(0, 1, len(vals))
    plt.plot(vals, y, label=label)

plt.xscale("log")
plt.xlabel("Valore (scala log)")
plt.ylabel("CDF")
plt.title("CDF delle principali metriche del workload")
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "cdf_metrics.png"))
plt.show()

print("Analisi incrociata completata. Figure e CSV salvati in 'figs/' e 'output/'")


# %% [markdown]
# ## 9) Recap finale

# %%
print("RECAP DEI PRINCIPALI RISULTATI")

recap_dict = {
    "Owner totali": n_owners,
    "Applicazioni totali": n_apps,
    "Funzioni totali": n_funcs,
    "Funzioni con dati di durata": n_fun_dur,
    "Funzioni senza dati di durata": n_fun_no_dur,
    "Applicazioni con dati di memoria": n_app_mem,
    "Applicazioni senza dati di memoria": n_app_no_mem,

    "Minuto di picco invocazioni": f"{peak_minute//60:02d}:{peak_minute%60:02d}",
    "Minuto minimo invocazioni": f"{min_minute//60:02d}:{min_minute%60:02d}",

    "Top 3 funzioni più invocate": top10_funcs["Function"].head(3).tolist(),
    "Top 3 applicazioni più invocate": top10_apps["App"].head(3).tolist(),

    "Top 3 funzioni più lente (mediana ms)": top_slowest_funcs.head(3).to_dict(),
    "Top 3 applicazioni più lente (mediana ms)": top_slowest_apps.head(3).to_dict(),

    "Distribuzione teorica stimata delle durate": dist.model["name"],

    "Top 10 applicazioni per memoria mediana (MB)": top10_apps_by_median.to_dict(),
    "Top 10 applicazioni per memoria P99 (MB)": top10_apps_by_pct99.to_dict(),

    "Numero medio di funzioni per applicazione": float(func_per_app.mean()),
    "Numero medio di trigger per applicazione": float(num_trig_per_app.mean()),

    "Dimensione campione analisi incrociata": len(sample_df)
}

for k, v in recap_dict.items():
    print(f"- {k}: {v}")

pd.DataFrame(
    [{"Metric": k, "Value": v} for k, v in recap_dict.items()]
).to_csv(
    os.path.join(OUTPUT_DIR, "recap_summary.csv"),
    index=False,
    sep=";",
    encoding="utf-8"
)

print("\nRecap finale salvato in output/recap_summary.csv")
