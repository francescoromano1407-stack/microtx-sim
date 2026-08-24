# Evidenza, provenienza e limiti

Le fonti machine-readable sono elencate in `data/provenance/sources.toml`. Le
configurazioni separano sempre:

- `CALIBRATED`: valore estratto, trasformato e validato per uno specifico
  estimando; la comparabilità richiede comunque unità, periodo e popolazione
  compatibili;
- `ANCHORED`: valore o meccanismo legato a una fonte, ma non ancora pienamente
  calibrato;
- `ILLUSTRATIVE`: assunzione di scenario;
- `SYNTHETIC`: valore generato per controllo strutturale o scenario artificiale,
  mai autorizzato automaticamente come stima scientifica.

La letteratura ufficiale disponibile giustifica prudenza causale. La risposta del
Governo britannico sulle loot box descrive associazioni robuste ma afferma che la
direzione causale non è stabilita; questo è precisamente il motivo del disegno a
mondi accoppiati. La definizione WHO di gaming disorder richiede compromissione
significativa e persistente: il simulatore registra proxy longitudinali, ma non
formula diagnosi cliniche.

I profili nazionali iniziali non devono essere confrontati come stime. ONS,
Statistics Korea e Statbel usano popolazioni, unità e periodi diversi. Gli anchor
nominali restano nella valuta locale: £36.663 e €31.299 sono memorizzati in
pence/centesimi prima della divisione mensile; KRW e JPY hanno unità minori a
esponente zero. Ogni anchor mensile viene poi mappato a `180000` *simulation
cents* mediante una scala **ILLUSTRATIVE**, non un cambio e non una PPP.

## Input usati in questa fase

| Gruppo | Uso corrente | Stato/limite |
|---|---|---|
| Età e forma del reddito | inizializzano direttamente i giocatori | pesi e dispersioni ILLUSTRATIVE |
| Reddito nominale ufficiale | anchor per la scala monetaria interna | non confrontabile tra Paesi |
| Reach, payer rate, spesa e deprivation | conservati come contratti | non ancora consumati dalle equazioni |
| Regole | consumate una per una dal regolatore | ogni regola ha fonte/status proprio; varie restano ILLUSTRATIVE |
| Audit | capacità e accuratezza operative | SYNTHETIC/config di scenario |
| Sussidi | domanda antecedente, eleggibilità domestica, proxy di design e contabilità | budget/pesi/residenza SYNTHETIC; rate e cap ufficiali non ancora calibrati |
| Evento carta | hazard consumato dal modello | prior ILLUSTRATIVE per analisi di sensibilità |

La popolazione copre le fasce configurate 8/10–69 anni e assegna per ora uguale
peso ai quattro Paesi: non rappresenta le rispettive popolazioni nazionali. La
categoria Statbel `65+` è soltanto un anchor descrittivo e non viene applicata
come se descrivesse esattamente i soli 65–69 anni.

Per una futura calibrazione il catalogo dovrà aggiungere snapshot immutabili,
tabella/cella o pagina, unità originale, trasformazione e checksum. Gli URL
`latest` sono utili per l'ancoraggio iniziale ma non bastano alla riproducibilità.
