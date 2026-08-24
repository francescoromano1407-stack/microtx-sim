# Specifica del modello

## Confine informativo

`World` contiene lo stato vero necessario al ricercatore. Nessuna policy riceve
un riferimento a `World`: le decisioni accettano viste immutabili costruite da
`InformationSystem`. Una vista riporta anche età del segnale, precisione attesa e
fonte. Informazioni migliori richiedono costo monetario, capacità analitica o
rischio legale.

La separazione è intenzionale:

```text
TruthStore -> sensore/report -> Observation -> Belief -> Decision
                 ^ costo, rumore, ritardo          |
                 +---------------------------------+
```

## Agenti

### Giocatori e nuclei familiari

I giocatori sono memorizzati in colonne NumPy per scalare a popolazioni grandi.
Ogni riga mantiene tratti continui, cinque motivazioni sovrapponibili, risorse
finanziarie e indicatori di funzionamento. Il comportamento dipende da questi
tratti e dall'osservazione disponibile, non da un'etichetta deterministica.

Per i minori si distinguono allowance, liquidità del nucleo, carta memorizzata,
limite di credito, supervisione e consenso. Un acquisto non autorizzato è un
evento raro stocastico con condizioni necessarie; può generare scoperta,
reclamo, rimborso e danno familiare.

### Aziende e giochi

Le aziende possiedono cassa, cultura di compliance, avversione al rischio,
capacità analitica e credenze sul mercato e sui controlli. Valutano candidati di
contenuto/monetizzazione tramite NPV percepito. Un nuovo contenuto competitivo
può superare il frontier stimato su un sottoinsieme delle statistiche, con almeno
un trade-off: non esiste un personaggio che domina tutto.

Il gioco è rappresentato da qualità, integrità competitiva, content cadence,
frontier multidimensionale, monetizzazione, popolazione attiva e ranking. Non
esistono personaggi, abilità o combattimenti codificati.

### Stati/regolatori

Ogni giurisdizione ha bilancio, capacità di controllo, priorità politiche,
regole, fondo sussidi e credenze sulla conformità delle imprese. Gli audit sono
eventi pianificati e leggono soltanto segnali/evidenze con sensibilità e
specificità finite. I sussidi possono essere condizionati a ricavi meno dipendenti
da spesa ad alto rischio e a metriche di accessibilità/qualità.

## Sistemi di mercato

Il ciclo logico è ordinato per evitare dipendenze accidentali:

1. shock esogeni e rinnovo dei budget;
2. produzione/acquisto di informazioni;
3. decisioni di aziende, contenuti e accordi bilaterali;
4. scelta del gioco e competizione astratta;
5. acquisti, stato comportamentale ed eventi rari;
6. popolarità vera e classifica pubblica rumorosa;
7. audit, sanzioni e sussidi;
8. contabilità e misure per il ricercatore.

Gli eventi lenti sono in una priority queue. Le decisioni dei giocatori sono
vettoriali. Aggregazioni per gioco/Stato usano `bincount`; le scelte dense sono
calcolate a blocchi, mantenendo il risultato esatto ma limitando la memoria.

