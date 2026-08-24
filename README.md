# Microtransazioni — simulazione agent-based causale

Questo repository contiene lo scheletro eseguibile di una simulazione agent-based
per studiare la domanda:

> Quanto danno aggiuntivo è causalmente attribuibile alle meccaniche di
> monetizzazione dei giochi mobile, al netto della vulnerabilità preesistente del
> giocatore, e quale combinazione di regolazione e finanziamento pubblico può
> sostenere un gioco economicamente viabile senza dipendere dalla spesa
> compulsiva?

Il progetto non simula il gameplay. Simula invece un mercato competitivo con
giocatori eterogenei, giochi, aziende e regolatori. Le categorie come casual,
competitivo e collezionista sono motivazioni continue e sovrapponibili; *whale* è
un segmento di spesa che emerge a posteriori, non un comportamento assegnato.

## Principi del modello

- **Conoscenza locale:** la verità latente è privata al motore. Gli agenti ricevono
  osservazioni rumorose, ritardate o acquistate e aggiornano credenze fallibili.
- **Causalità esplicita:** mondi gemelli possono condividere popolazione e shock
  esogeni, variando soltanto monetizzazione, regolazione o sussidi.
- **Eterogeneità:** reddito, età, motivazioni, vulnerabilità, skill, controllo
  parentale, alfabetizzazione finanziaria e accesso al credito sono continui e
  correlati.
- **Comportamento emergente:** contenuti, intensità di monetizzazione, elusione,
  collaborazione e collusione sono scelti tramite utilità/NPV percepito; non
  seguono copioni.
- **Contabilità esatta:** denaro in *simulation cents* interi, aggregazioni esatte
  e scelte tra giochi calcolate a blocchi senza campionare alternative. Le cifre
  nominali GBP/KRW/JPY/EUR restano separate e non sono trattate come cambi o PPP.
- **Provenienza obbligatoria:** una campagna scientifica deve rifiutare parametri
  sintetici o non validati. La configurazione `smoke.toml` serve soltanto ai test.

## Stato attuale

Questa fase costruisce lo scheletro e una breve esecuzione di controllo, non una
campagna. I profili iniziali UK/KR/JP/BE sono illustrativi e ancorati a fonti
ufficiali, ma non costituiscono ancora una calibrazione comparabile tra Paesi.
Questa distinzione è verificata dal codice.

`configs/base.toml` descrive la scala futura (50.000 giocatori, 5 aziende e 8
giochi), ma è deliberatamente bloccato: contiene dipendenze non calibrate e non
può produrre stime scientifiche. `configs/smoke.toml` è l'unico scenario
eseguibile in questa fase e usa 384 giocatori per tre cicli.

## Componenti principali

- `core/world.py`: unica sede della verità latente e orchestrazione degli eventi;
- `systems/player_dynamics.py`: scelta esatta tra alternative note, competizione
  astratta, acquisti, credito, evento carta e sette esiti di danno separati;
- `systems/firm_strategy.py`: aggiornamenti, monetizzazione, ricerca, compliance,
  acquisizione, collaborazione, collusione, elusione e domanda di sussidio;
- `systems/regulation.py`: selezione dei controlli da segnali locali e risoluzione
  con sensibilità/specificità finite;
- `causal/paired_worlds.py`: mondi gemelli con popolazione e shock esogeni comuni;
- `data/provenance/sources.toml`: registro delle fonti e del loro ambito.

## Avvio locale

```text
python -m pip install -e .
python -m microtx_sim validate configs/smoke.toml
python -m microtx_sim smoke configs/smoke.toml
```

Lo smoke test esegue pochi cicli e produce soltanto un riepilogo in memoria. Per
architettura, assunzioni e disegno causale si vedano `docs/`.
