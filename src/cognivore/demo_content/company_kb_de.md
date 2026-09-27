# Wissensbasis von Falke Cloud (Demo-Dokument)

## Über das Unternehmen
Falke Cloud bietet Video-Hosting und eine Streaming-API für
Online-Lernplattformen. Das Unternehmen betreut 170 Geschäftskunden in
Deutschland, Österreich und der Schweiz. Hauptsitz ist Berlin, die
Rechenzentren stehen in Berlin, Frankfurt und Zürich.

## Tarife und Preise
- **Start-Tarif**: 25 € pro Monat. 50 GB Speicher, 500 GB Datenvolumen pro
  Monat, bis zu 5 Nutzer. Support per E-Mail, Antwort innerhalb von 48 Stunden.
- **Business-Tarif**: 129 € pro Monat. 1 TB Speicher, 5 TB Datenvolumen, bis
  zu 50 Nutzer. Priorisierter Support mit Antwort innerhalb von 4 Stunden und
  API-Zugang für den automatischen Upload von Videos.
- **Enterprise-Tarif**: Preis auf Anfrage. Unbegrenzter Speicher, dediziertes
  Rechenzentrum nach Wahl, persönlicher Ansprechpartner, SLA von 99,95 % und
  Reaktion auf kritische Störungen innerhalb von 30 Minuten.

Ein Tarifwechsel gilt sofort; die Differenz für die restlichen Tage des
Abrechnungszeitraums wird anteilig verrechnet.

## SLA und Verfügbarkeit
Garantiert werden 99,9 % Verfügbarkeit im Start- und Business-Tarif und
99,95 % im Enterprise-Tarif. Bei einer SLA-Verletzung erhält der Kunde für
jede Stunde Ausfall über dem erlaubten Rahmen eine Gutschrift von 5 % der
Monatsgebühr, höchstens 100 % pro Monat. Geplante Wartung findet mittwochs
von 02:00 bis 04:00 Uhr (Berliner Zeit) statt und zählt nicht als Ausfall,
wenn sie mindestens 72 Stunden vorher angekündigt wurde.

## Sicherheit und Compliance
Alle Videos werden im Ruhezustand (AES-256) und bei der Übertragung
(TLS 1.3) verschlüsselt. Falke Cloud ist seit 2025 nach ISO 27001
zertifiziert. Backups laufen alle 6 Stunden, Wiederherstellungspunkte werden
30 Tage aufbewahrt. Mitarbeitende haben ohne ausdrückliche Freigabe des
Kunden über ein Support-Ticket keinen Zugriff auf Videoinhalte.

## Integrationen
Webhooks gibt es für: Upload abgeschlossen, Transkodierung abgeschlossen,
Video gelöscht, Datenvolumen überschritten. Fertige Integrationen: Zoom
Cloud Recording, Moodle (LTI-1.3-Plugin), Google Classroom und Microsoft
Teams. Die REST-API ist mit OpenAPI 3.0 dokumentiert und im Business- und
Enterprise-Tarif verfügbar.

## API-Limits
Business-Tarif: 600 Anfragen pro Minute, höchstens 5 GB pro Upload (größere
Dateien per Multipart-Upload). Enterprise-Tarif: individuell vereinbart,
üblicherweise ab 3000 Anfragen pro Minute. Bei Überschreitung antwortet die
API mit HTTP 429 und dem Header Retry-After.

## Rückerstattungsrichtlinie
Innerhalb von 14 Tagen nach der ersten Zahlung gibt es eine volle
Rückerstattung, sofern der Kunde nicht mehr als 10 GB Datenvolumen genutzt
hat. Nach 14 Tagen erfolgt keine Rückerstattung, der Kunde kann aber in
einen kleineren Tarif wechseln oder das Abonnement bis zu 3 Monate pausieren,
ohne gespeicherte Videos zu verlieren.

## Häufige Fragen
1. „Mein Video wird sehr langsam verarbeitet“: Die Transkodierung eines
   10-minütigen 1080p-Videos dauert im Start- und Business-Tarif im Schnitt
   4-6 Minuten, im Enterprise-Tarif bis zu 90 Sekunden.
2. „Wie exportiere ich Untertitel?“: Untertitel werden automatisch erzeugt
   und können im Tab „Untertitel“ als SRT oder VTT heruntergeladen werden.
