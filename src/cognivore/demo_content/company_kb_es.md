# Base de conocimiento de Cóndor Cloud (documento de demostración)

## Sobre la empresa
Cóndor Cloud es un proveedor de alojamiento de vídeo y API de streaming para
plataformas de formación en línea. Atiende a 160 clientes empresariales en
España y América Latina. La sede está en Madrid y los centros de datos se
encuentran en Madrid, Ciudad de México y São Paulo.

## Planes y precios
Cóndor Cloud ofrece tres planes; los precios son mensuales y sin permanencia.
- **Básico**: 25 € al mes. 50 GB de almacenamiento, 500 GB de tráfico mensual,
  hasta 5 usuarios. Soporte por correo electrónico con respuesta en 48 horas.
- **Profesional**: 129 € al mes. 1 TB de almacenamiento, 5 TB de tráfico,
  hasta 50 usuarios. Soporte prioritario con respuesta en 4 horas y acceso a
  la API para subir vídeos de forma automática.
- **Empresa**: precio a medida. Almacenamiento ilimitado, centro de datos
  dedicado a elección del cliente, gestor de cuenta personal, SLA del 99,95 %
  y respuesta a incidentes críticos en 30 minutos.

El cambio de plan es inmediato; la diferencia de los días restantes del
periodo se calcula de forma proporcional.

## SLA y fiabilidad
La disponibilidad garantizada es del 99,9 % en los planes Básico y
Profesional y del 99,95 % en el plan Empresa. Si se incumple el SLA, el
cliente recibe un crédito del 5 % de la cuota mensual por cada hora de
inactividad por encima de lo permitido, con un máximo del 100 % al mes. El
mantenimiento programado se realiza los miércoles de 02:00 a 04:00 (hora de
Madrid) y no cuenta como inactividad si se avisa con al menos 72 horas.

## Seguridad y cumplimiento
Todos los vídeos se cifran en reposo (AES-256) y en tránsito (TLS 1.3).
Cóndor Cloud obtuvo la certificación ISO 27001 en 2025. Las copias de
seguridad se hacen cada 6 horas y los puntos de restauración se guardan 30
días. El personal no tiene acceso al contenido de los vídeos sin la
autorización expresa del cliente mediante un ticket de soporte.

## Integraciones
Hay webhooks para: subida completada, transcodificación completada, vídeo
eliminado y límite de tráfico superado. Integraciones listas: Zoom Cloud
Recording, Moodle (complemento LTI 1.3), Google Classroom y Microsoft Teams.
La API REST está documentada en OpenAPI 3.0 y disponible en los planes
Profesional y Empresa.

## Límites de la API
Plan Profesional: 600 solicitudes por minuto y 5 GB como máximo por subida
(los archivos mayores usan subida por partes). Plan Empresa: límites
negociados, normalmente desde 3000 solicitudes por minuto. Al superar el
límite la API devuelve HTTP 429 con la cabecera Retry-After.

## Política de reembolso
Se ofrece un reembolso completo dentro de los 14 días siguientes al primer
pago, siempre que el cliente no haya usado más de 10 GB de tráfico. Pasados
14 días no hay reembolso, pero el cliente puede bajar de plan o pausar la
suscripción hasta 3 meses sin perder los vídeos guardados.

## Preguntas frecuentes
1. «El vídeo tarda mucho en procesarse»: la transcodificación de un vídeo de
   10 minutos en 1080p tarda de media 4-6 minutos en Básico y Profesional, y
   hasta 90 segundos en Empresa.
2. «¿Cómo exporto los subtítulos?»: se generan automáticamente y se pueden
   descargar en formato SRT y VTT desde la pestaña «Subtítulos».
