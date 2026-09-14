# Skylark Cloud Knowledge Base (demo document)

## About the company
Skylark Cloud is a video-hosting and streaming API provider for online
learning platforms. The company serves 210 business customers across North
America and the EU. Headquarters are in Austin, TX, with data centers in
Austin, Dublin, and Singapore.

## Pricing plans
- **Starter** -- $29/month. 50 GB storage, 500 GB of monthly bandwidth, up
  to 5 team seats. Email support, response within 48 hours.
- **Growth** -- $149/month. 1 TB storage, 5 TB bandwidth, up to 50 team
  seats. Priority support, response within 4 hours, API access for
  automated video uploads.
- **Enterprise** -- custom pricing. Unlimited storage, a dedicated data
  center of the customer's choosing, a named account manager, a 99.95% SLA,
  and a 30-minute response target for critical incidents.

Plan changes take effect immediately; the price difference for the
remaining days in the billing period is charged or credited on a
pro-rated basis.

## SLA and reliability
Guaranteed uptime is 99.9% on the Starter and Growth plans, 99.95% on
Enterprise. An SLA breach credits the customer 5% of that month's fee for
every hour of downtime beyond the allowance, capped at a 100% refund for
the month. Scheduled maintenance runs Wednesdays 02:00-04:00 UTC and does
not count as downtime as long as customers were notified at least 72 hours
in advance.

## Security and compliance
All video is encrypted at rest (AES-256) and in transit (TLS 1.3). Skylark
Cloud completed SOC 2 Type II certification in 2025. Backups run every 6
hours and restore points are kept for 30 days. Customer data is accessible
only on written request and only to that customer -- Skylark Cloud staff
have no standing access to video content without the customer's explicit
sign-off through a support ticket.

## Integrations
Webhooks are available for: upload completed, transcoding completed, video
deleted, and bandwidth-limit exceeded. Ready-made integrations include Zoom
Cloud Recording (automatic import of recordings), Moodle (an LTI 1.3
plugin), Google Classroom, and Slack. The REST API is documented as
OpenAPI 3.0 and is available to customers on the Growth and Enterprise
plans.

## API limits
Growth plan: 600 requests/minute, 5 GB max per single upload request
(larger files use multipart upload). Enterprise plan: limits are
negotiated individually, typically starting at 3,000 requests/minute.
Exceeding the limit returns HTTP 429 with a Retry-After header.

## Refund policy
A full refund is available within 14 days of the first payment, provided
the customer has used no more than 10 GB of bandwidth. After 14 days,
refunds are not issued, but the customer may downgrade their plan or pause
the subscription for up to 3 months without losing any stored video.

## Frequently asked support questions
1. "My video is taking a long time to process" -- average transcoding time
   for a 10-minute 1080p video is 4-6 minutes on Starter/Growth, and up to
   90 seconds on Enterprise (dedicated transcoding capacity).
2. "How do I export subtitles?" -- subtitles are generated automatically
   (speech recognition) and are available in SRT and VTT formats under the
   video's "Subtitles" tab.
3. "Can I restrict video playback by IP or domain?" -- yes, on the Growth
   plan and above, video settings support an allow-list of viewer email
   domains and a list of allowed countries.
