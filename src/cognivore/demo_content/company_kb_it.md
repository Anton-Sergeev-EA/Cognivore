# Base di conoscenza di Airone Cloud (documento dimostrativo)

## L’azienda
Airone Cloud offre hosting video e un’API di streaming per le piattaforme di
formazione online. Serve 120 clienti aziendali in Italia, Svizzera e Malta.
La sede è a Milano e i data center si trovano a Milano, Roma e Francoforte.

## Piani e prezzi
Airone Cloud propone tre piani; i prezzi sono mensili e senza vincoli.
- **Base**: 25 € al mese. 50 GB di spazio, 500 GB di traffico mensile, fino a
  5 utenti. Assistenza via e-mail con risposta entro 48 ore.
- **Professionale**: 129 € al mese. 1 TB di spazio, 5 TB di traffico, fino a
  50 utenti. Assistenza prioritaria con risposta entro 4 ore e accesso
  all’API per il caricamento automatico dei video.
- **Impresa**: prezzo su preventivo. Spazio illimitato, data center dedicato a
  scelta del cliente, account manager personale, SLA del 99,95% e intervento
  entro 30 minuti per gli incidenti critici.

Il cambio di piano è immediato; la differenza per i giorni rimanenti del
periodo viene calcolata in proporzione.

## SLA e affidabilità
La disponibilità garantita è del 99,9% per i piani Base e Professionale e del
99,95% per il piano Impresa. In caso di mancato rispetto dello SLA il cliente
riceve un credito del 5% del canone mensile per ogni ora di fermo oltre la
soglia consentita, fino al 100% al mese. La manutenzione programmata si
svolge il mercoledì dalle 02:00 alle 04:00 (ora di Roma) e non conta come
fermo se il cliente è avvisato almeno 72 ore prima.

## Sicurezza e conformità
Tutti i video sono cifrati a riposo (AES-256) e in transito (TLS 1.3). Airone
Cloud ha ottenuto la certificazione ISO 27001 nel 2025. I backup vengono
eseguiti ogni 6 ore e i punti di ripristino sono conservati per 30 giorni. Il
personale non accede ai contenuti dei video senza l’autorizzazione esplicita
del cliente tramite un ticket di assistenza.

## Integrazioni
Sono disponibili webhook per: caricamento completato, transcodifica
completata, video eliminato, limite di traffico superato. Integrazioni
pronte: Zoom Cloud Recording, Moodle (plugin LTI 1.3), Google Classroom e
Microsoft Teams. L’API REST è documentata in OpenAPI 3.0 ed è disponibile nei
piani Professionale e Impresa.

## Limiti dell’API
Piano Professionale: 600 richieste al minuto, massimo 5 GB per caricamento
(i file più grandi usano il caricamento multiparte). Piano Impresa: limiti
concordati, di solito da 3000 richieste al minuto. Oltre il limite l’API
restituisce HTTP 429 con l’intestazione Retry-After.

## Politica di rimborso
Il rimborso completo è possibile entro 14 giorni dal primo pagamento, se il
cliente non ha usato più di 10 GB di traffico. Dopo 14 giorni non sono
previsti rimborsi, ma il cliente può passare a un piano inferiore o
sospendere l’abbonamento fino a 3 mesi senza perdere i video salvati.

## Domande frequenti
1. «Il video viene elaborato lentamente»: la transcodifica di un video di 10
   minuti in 1080p richiede in media 4-6 minuti con Base e Professionale e
   fino a 90 secondi con Impresa.
2. «Come esporto i sottotitoli?»: i sottotitoli sono generati automaticamente
   e si scaricano in formato SRT e VTT dalla scheda «Sottotitoli».
