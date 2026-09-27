# Base de connaissances Héron Cloud (document de démonstration)

## À propos de l’entreprise
Héron Cloud fournit l’hébergement vidéo et une API de streaming aux
plateformes de formation en ligne. L’entreprise compte 190 clients
professionnels en France, en Belgique, en Suisse et au Canada. Le siège est
à Lyon ; les centres de données sont situés à Lyon, Paris et Montréal.

## Offres et tarifs
Héron Cloud propose trois offres ; les tarifs sont mensuels et sans engagement.
- **Essentiel** : 25 € par mois. 50 Go de stockage, 500 Go de trafic mensuel,
  jusqu’à 5 utilisateurs. Support par e-mail, réponse sous 48 heures.
- **Pro** : 129 € par mois. 1 To de stockage, 5 To de trafic, jusqu’à 50
  utilisateurs. Support prioritaire, réponse sous 4 heures, accès à l’API
  pour l’envoi automatique des vidéos.
- **Entreprise** : tarif sur devis. Stockage illimité, centre de données
  dédié au choix du client, responsable de compte attitré, SLA de 99,95 % et
  intervention sous 30 minutes en cas d’incident critique.

Le changement d’offre est immédiat ; la différence pour les jours restants
de la période est calculée au prorata.

## SLA et fiabilité
La disponibilité garantie est de 99,9 % pour les offres Essentiel et Pro et
de 99,95 % pour l’offre Entreprise. En cas de non-respect du SLA, le client
reçoit un avoir de 5 % de l’abonnement mensuel par heure d’indisponibilité
au-delà du seuil autorisé, dans la limite de 100 % par mois. La maintenance
planifiée a lieu le mercredi de 02:00 à 04:00 (heure de Paris) et ne compte
pas comme indisponibilité si le client est prévenu au moins 72 heures avant.

## Sécurité et conformité
Toutes les vidéos sont chiffrées au repos (AES-256) et en transit (TLS 1.3).
Héron Cloud a obtenu la certification ISO 27001 en 2025. Les sauvegardes ont
lieu toutes les 6 heures et les points de restauration sont conservés 30
jours. Le personnel n’accède pas au contenu des vidéos sans l’accord
explicite du client via un ticket de support.

## Intégrations
Des webhooks sont disponibles pour : envoi terminé, transcodage terminé,
vidéo supprimée, limite de trafic dépassée. Intégrations prêtes à l’emploi :
Zoom Cloud Recording, Moodle (plugin LTI 1.3), Google Classroom et Microsoft
Teams. L’API REST est documentée en OpenAPI 3.0 et accessible avec les
offres Pro et Entreprise.

## Limites de l’API
Offre Pro : 600 requêtes par minute, 5 Go maximum par envoi (les fichiers
plus volumineux utilisent l’envoi en plusieurs parties). Offre Entreprise :
limites négociées, généralement à partir de 3000 requêtes par minute. Au-delà
de la limite, l’API renvoie HTTP 429 avec l’en-tête Retry-After.

## Politique de remboursement
Un remboursement intégral est possible dans les 14 jours suivant le premier
paiement, à condition que le client n’ait pas consommé plus de 10 Go de
trafic. Après 14 jours, aucun remboursement n’est effectué, mais le client
peut passer à une offre inférieure ou suspendre son abonnement jusqu’à 3
mois sans perdre ses vidéos.

## Questions fréquentes
1. « Ma vidéo met longtemps à être traitée » : le transcodage d’une vidéo de
   10 minutes en 1080p prend en moyenne 4 à 6 minutes avec Essentiel et Pro,
   et jusqu’à 90 secondes avec Entreprise.
2. « Comment exporter les sous-titres ? » : ils sont générés automatiquement
   et téléchargeables aux formats SRT et VTT depuis l’onglet « Sous-titres ».
