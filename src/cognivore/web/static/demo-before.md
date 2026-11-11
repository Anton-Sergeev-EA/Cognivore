# Telemetry gateway specification, revision 1

## Power
PWR-1: The gateway shall provide backup power for at least 30 minutes.
PWR-2: The gateway must send a power-loss alarm within 5 seconds.

## Storage
The gateway must retain event logs for at least 90 days.
LOG-2: The gateway must archive raw diagnostics during normal operation.

## Networking
NET-1: The gateway may retry upload when the network is congested.
NET-2: The gateway must encrypt telemetry in transit.
