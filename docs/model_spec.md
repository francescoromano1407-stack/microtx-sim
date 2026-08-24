# Specifica del modello

## Confine informativo

`World` contiene lo stato vero necessario al ricercatore. Nessuna policy riceve
un riferimento a `World`: le decisioni accettano viste immutabili costruite da
sistemi specializzati. I segnali riportano età, precisione attesa e fonte; le
aziende possono acquistare ricerca e i regolatori ottengono la verità di
compliance soltanto attraverso un audit imperfetto. Le classifiche sono ritardate
e rumorose. La media latente dei danni non viene consegnata allo Stato: i suoi
indicatori derivano da reclami e anomalie osservate.

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
tratti e dall'osservazione disponibile, non da un'etichetta deterministica. La
qualità del gioco entra tramite esperienza personale rumorosa; scoperta e scelta
tra titoli usano soltanto prezzi/meccaniche pubblicate e classifiche pubbliche.

Per i minori si distinguono allowance, liquidità del nucleo, carta memorizzata,
limite di credito, supervisione e consenso. Un acquisto non autorizzato è un
evento raro stocastico con condizioni necessarie e un tetto per nucleo; alimenta
l'esito separato di spesa non autorizzata e può generare un reclamo osservabile.
Rimborsi e contenzioso sono estensioni future, non risultati correnti.

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
da spesa ad alto rischio e a metriche di accessibilità/qualità. Nello scheletro,
ogni azienda ha una giurisdizione domestica sintetica, la domanda deve precedere
la revisione e lo Stato usa un proxy verificabile di sicurezza del design: non
gli viene rivelata la quota latente di ricavo unsafe del ricercatore.

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

La telemetria aziendale usa flussi dall'ultima decisione, non ricavi cumulativi.
Gli shock di scelta hanno una coordinata fissa per azione anche quando l'azione
non è finanziariamente ammissibile; questo mantiene lo stesso significato nei
mondi gemelli. Un limite sperimentale a una meccanica resta attivo dopo
aggiornamenti e collusioni successive.

Il ledger registra flussi di cassa a partita doppia. Interessi maturati e multe
accertate ma non riscosse restano passività separate; il margine aziendale
sottrae le multe residue, mentre la cassa mostra soltanto gli importi riscossi.

Gli eventi lenti sono in una priority queue. Le decisioni dei giocatori sono
vettoriali. Le somme monetarie usano scatter-add interi; le scelte dense sono
calcolate a blocchi, mantenendo tutte le alternative note ma limitando la memoria.

## Costo computazionale

Con `P` giocatori, `G` giochi, `F` aziende, `S` Stati e blocco `B`, il ciclo dei
giocatori costa `O(P·G)` e usa memoria temporanea `O(B·G + P)`: il blocco cambia
solo l'ordine del calcolo, non campiona né elimina alternative. Popolarità e
contabilità costano `O(P + G)`; gli audit `O(S·F log F)`. La ricerca esatta dei
contenuti valuta tutti i sottoinsiemi propri delle `D` statistiche e la griglia
di boost, ma avviene solo agli aggiornamenti e `D` è limitato a 12.

Gli intervalli di calendario devono essere multipli esatti di `tick_days`; una
configurazione disallineata viene rifiutata anziché spostare silenziosamente gli
eventi.

“Senza approssimazione” riguarda quindi il calcolo delle alternative e del
denaro. Le equazioni comportamentali restano ipotesi di modello stocastiche e
devono essere calibrate: non sono verità psicologiche né diagnosi cliniche.
