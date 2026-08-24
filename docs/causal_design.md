# Disegno causale

## Estimando principale

Per il giocatore `i`, il danno incrementale di un regime di monetizzazione `M`
rispetto a un regime neutrale `M0` è:

```text
tau_i = H_i(M, R, S; U_i) - H_i(M0, R, S; U_i)
```

`U_i` contiene vulnerabilità e shock esogeni pre-trattamento condivisi. `R` è la
regolazione e `S` il finanziamento pubblico. Le componenti di `H` restano
separate: stress finanziario, spesa oltre budget, debito, perdita di controllo,
regret/rimborsi, tempo e compromissione del funzionamento.

La vulnerabilità non viene sottratta con una regressione a posteriori: è mantenuta
identica in mondi gemelli tramite un generatore controbasato indicizzato da seed,
ciclo, meccanismo ed entità. In questo modo un ramo che compie più acquisti non
sposta casualmente gli shock futuri dell'altro ramo.

## Interferenza

Ranking, passaparola e reazioni delle aziende violano l'ipotesi di non interferenza
tra individui. Il risultato principale è quindi un **effetto di regime in
equilibrio di mercato**, non un ATE individuale ingenuo. Effetti diretti e spillover
richiedono randomizzazione a cluster (gioco o giurisdizione) o esposizioni di rete
esplicitamente registrate.

## Esperimento fattoriale futuro

Il runner supporta una matrice `monetizzazione x regolazione x sussidio`. Gli
outcome di viabilità comprendono solvibilità, margine, continuità degli
aggiornamenti e quota di ricavi non proveniente da sessioni classificate ad alto
rischio. Nessuna campagna viene eseguita in questa fase.

