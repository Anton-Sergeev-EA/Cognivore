# Telemetry gateway specification, revision 2

## Networking
NET-2: The gateway must encrypt telemetry in transit.
NET-1: The gateway must retry upload only if the network is congested.
NET-3: The gateway must authenticate each remote client.

## Storage
The gateway must keep event logs for at least 180 days.

## Power
PWR-2: The gateway must send a power-loss alarm within 5 seconds.
PWR-1: The gateway shall provide reserve power for at least 20 minutes.
