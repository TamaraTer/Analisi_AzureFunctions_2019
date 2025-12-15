# Analisi Dataset Azure Functions 2019

Questo progetto realizza un’analisi completa del dataset pubblico Azure Functions 2019, pubblicato da Microsoft e utilizzato nel paper “Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider” (USENIX ATC 2020). 

Il progetto analizza un workload serverless a partire da tre grandi dataset (invocations, durations e memory), studiando:

- il volume e l’andamento temporale delle invocazioni,
- le durate di esecuzione delle funzioni,
- la memoria allocata dalle applicazioni,
- le relazioni incrociate tra carico, prestazioni e utilizzo di memoria,
- la struttura delle applicazioni tramite la ricostruzione delle catene di funzioni
  e dei trigger associati.

L’analisi è interamente sviluppata in Python utilizzando PyArrow, Pandas, NumPy, Matplotlib e Seaborn. Per garantire la scalabilità su macchine con RAM limitata, i dataset CSV vengono convertiti automaticamente in formato Parquet e processati in streaming tramite batch.

In aggiunta all’analisi descrittiva, il progetto include una stima automatica della distribuzione teorica delle durate medie delle funzioni. Diverse distribuzioni candidate vengono confrontate tramite il test di Kolmogorov–Smirnov, al fine di individuare il modello che approssima meglio i dati osservati (nel caso analizzato, una distribuzione lognormale).

## Dataset

Il dataset originale non è incluso nel repository per motivi di dimensione ma può essere scaricato dal link ufficiale Microsoft:  
https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz

Dopo aver estratto l’archivio, è sufficiente copiare le cartelle `invocations`, `durations` e `memory` all’interno della directory `data/` del progetto.  
Il codice rileverà automaticamente i file e procederà alla conversione e analisi.


## Struttura del dataset originale

Il progetto utilizza il dataset pubblico **Azure Functions Trace 2019**, rilasciato da Microsoft.  
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
- 12 file CSV (`app_memory_percentiles.anon.d01.csv` … `d12.csv`)
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

Per avviare il progetto è consigliabile utilizzare un ambiente virtuale.

Le dipendenze si installano con:  
- `pip install -r requirements.txt`

L’analisi può essere eseguita avviando il notebook Jupyter oppure lo script Python. 

Durante l’esecuzione il codice converte automaticamente i CSV in Parquet, analizza i tre dataset in modalità streaming, produce grafici (memoria, invocazioni, durate, trigger, serie temporali, distribuzioni log-log, CDF, analisi incrociate) e salva tutte le statistiche e i risultati nelle cartelle dedicate.

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
- analisi della concatenazione delle funzioni,
- analisi incrociata invocations–durations–memory,
- grafici,
- file CSV riepilogativi.

## Licenza del dataset

Il dataset Azure Functions 2019 è rilasciato da Microsoft con licenza CC-BY.  
Per l’attribuzione corretta, si rimanda al paper originale USENIX ATC 2020.

## Contatti

Per informazioni, chiarimenti o approfondimenti sul progetto:
**Tamara Tersigni – tamara.tersigni@studenti.unipg.it**


