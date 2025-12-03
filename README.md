# Analisi Dataset Azure Functions 2019

- Questo progetto realizza un’analisi completa del dataset pubblico Azure Functions 2019, pubblicato da Microsoft e utilizzato nel paper “Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider” (USENIX ATC 2020). 
- L’analisi include lo studio delle invocazioni delle funzioni serverless, delle durate di esecuzione, della memoria allocata e delle relazioni incrociate tra questi insiemi di dati. Tutte le elaborazioni sono sviluppate in Python utilizzando PyArrow, Pandas, NumPy, Matplotlib e Seaborn, con un’attenzione particolare alla scalabilità: il progetto è infatti progettato per funzionare anche su macchine con RAM limitata grazie alla lettura in streaming a batch e alla conversione automatica dei CSV in formato Parquet.

## Dataset

- Il dataset originale non è incluso per motivi di dimensione.
- Può essere scaricato dal link ufficiale Microsoft (https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz). 
- Dopo aver estratto l’archivio, è sufficiente copiare le cartelle “invocations”, “durations” e “memory” all’interno della directory “data/” del progetto. Il codice rileverà automaticamente la presenza dei file e procederà alla conversione e analisi.

## Struttura del dataset originale

Il progetto utilizza una versione ridotta del dataset pubblico **Azure Functions Trace 2019**, rilasciato da Microsoft.  
Il dataset contiene informazioni reali sull’utilizzo di Azure Functions, raccolte per 14 giorni nel 2019.

I dati sono suddivisi in tre insiemi principali:

### 1. Invocations (invocazioni delle funzioni)
- 14 file CSV, uno per ogni giorno (`invocations_per_function_md.anon.d01.csv` … `d14.csv`)
- Ogni file contiene:
  - `HashOwner`: ID del proprietario dell’applicazione  
  - `HashApp`: ID dell’applicazione  
  - `HashFunction`: ID della funzione  
  - `Trigger`: tipo di trigger  
  - `1 … 1440`: numero di invocazioni per minuto (1440 colonne = 24h × 60 min)

### 2. Durations (durata delle esecuzioni)
- 14 file CSV (`function_durations_percentiles.anon.d01.csv` … `d14.csv`)
- Contengono:
  - durata media, minima e massima  
  - numero di esecuzioni  
  - percentili della durata media (weighted)

### 3. Memory (memoria allocata)
- 12 file CSV (`app_memory_percentiles.anon.d01.csv` … d12.csv)
- Per ogni applicazione:
  - memoria media allocata (in MB)
  - percentili dell’allocazione media su base minuto

---

## Struttura del progetto

```
Analisi_AzureFunctions_2019/
│
├── data/                         → contiene i CSV originali e le sottocartelle
│   ├── invocations/              → (vuota all’inizio)
│   ├── durations/                → (vuota all’inizio)
│   └── memory/                   → (vuota all’inizio)
│
├── parquet/                      → generata automaticamente (CSV convertiti)
├── figs/                         → grafici prodotti dall’analisi
├── output/                       → risultati numerici, CSV riassuntivi
│
├── analisi_serverless.ipynb      → notebook principale
├── analisi_serverless.py         → versione script Python
└── requirements.txt              → dipendenze
```

## Esecuzione

- Per avviare il progetto è consigliabile utilizzare un ambiente virtuale
- Le dipendenze si installano con `pip install -r requirements.txt`. 
- L’analisi può essere eseguita avviando il notebook Jupyter oppure lo script Python. 
- Durante l’esecuzione il codice converte automaticamente i CSV in Parquet, analizza i tre dataset in modalità streaming, produce grafici (memoria, invocazioni, durate, trigger, serie temporali, distribuzioni log-log, CDF, analisi incrociate) e salva tutte le statistiche e i risultati nelle cartelle dedicate.

## Note sulla scalabilità e performance

Il codice è stato progettato tenendo conto del fatto che il dataset originale è molto grande (in particolare le invocations, con 1440 colonne di minuti per file).
Per permettere l’esecuzione anche su PC meno potenti, il progetto utilizza:

- PyArrow Dataset (lettura a batch)
- Conversione CSV → Parquet
- Aggregazioni incrementali
- Nessun caricamento completo in memoria


## Output generati

Il progetto genera:
- statistiche sulle invocazioni,
- distribuzioni delle durate (Average, Min, Max),
- analisi della memoria allocata,
- rilevamento anomalie,
- grafici log-log, CDF, serie temporali,
- analisi incrociata invocations–durations–memory,
- file CSV riepilogativi.

## Licenza del dataset

Il dataset Azure Functions 2019 è rilasciato da Microsoft con licenza CC-BY.  
Per eventuali utilizzi accademici è necessario citare il paper USENIX ATC 2020.

## Contatti

Per informazioni, chiarimenti o approfondimenti sul progetto:
**Tamara Tersigni – tamara.tersigni@studenti.unipg.it**


