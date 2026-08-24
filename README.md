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
- **Comportamento emergente:** contenuti, prezzi, elusione, collaborazione e
  collusione sono scelti tramite utilità/NPV percepito; non seguono copioni.
- **Contabilità esatta:** denaro in centesimi interi, aggregazioni esatte e scelte
  tra giochi calcolate a blocchi senza campionare alternative.
- **Provenienza obbligatoria:** una campagna scientifica deve rifiutare parametri
  sintetici o non validati. La configurazione `smoke.toml` serve soltanto ai test.

## Stato attuale

Questa fase costruisce lo scheletro e una breve esecuzione di controllo, non una
campagna. I profili iniziali IT/DE/UK sono illustrativi e ancorati a fonti
ufficiali, ma non costituiscono ancora una calibrazione comparabile tra Paesi.
Questa distinzione è verificata dal codice.

## Avvio locale

```text
python -m pip install -e .
python -m microtx_sim validate configs/smoke.toml
python -m microtx_sim smoke configs/smoke.toml
```

Lo smoke test esegue pochi cicli e produce soltanto un riepilogo in memoria. Per
architettura, assunzioni e disegno causale si vedano `docs/`.

