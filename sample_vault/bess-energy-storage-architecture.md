---
title: "Battery Energy Storage Systems (BESS) Architecture & Telemetry"
tags: ["iot", "energy-storage", "bess", "modbus", "edge-computing"]
---

# Battery Energy Storage Systems (BESS) Architecture & Telemetry

Commercial and Industrial (C&I) Battery Energy Storage Systems (BESS) operate as distributed energy assets that buffer renewable generation, arbitrate peak electricity pricing, and provide fast frequency response to the electrical grid.

## 1. System Hardware Layers
A typical enterprise BESS installation comprises three distinct control tiers:
- **Battery Racks & BMS (Battery Management System)**: The low-level embedded hardware monitoring individual lithium-ion cell voltages, temperatures, and state-of-charge (SoC). The BMS enforces hardware safety cutoffs for over-voltage, under-voltage, thermal runaway, and insulation resistance faults.
- **Power Conversion System (PCS / Inverter)**: The bi-directional 4-quadrant inverter converting DC battery power to AC 480V/grid power and vice versa. It controls real power ($P$, measured in kW) and reactive power ($Q$, measured in kVAR) with sub-second response times.
- **Site Edge Controller (EMS - Energy Management System)**: An industrial Linux embedded gateway (running on ARM or x86 hardware) communicating locally with PCS and BMS via industrial protocols such as **Modbus TCP/RTU** and **CANbus**.

## 2. Modbus Protocol Integration & Register Polling
Communication between the edge gateway and power electronics predominantly utilizes the **Modbus TCP** protocol (port 502) or RS-485 **Modbus RTU**:
- **Holding Registers (Function Code 03/16)**: Used for bidirectional control setpoints, such as `Active_Power_Setpoint` (kW), `Power_Factor_Target`, and `Inverter_State_Command` (Start/Stop/Fault Reset).
- **Input Registers (Function Code 04)**: Used for read-only sensor telemetry, including `AC_Frequency`, `DC_Voltage`, `Rack_SOC_Percentage`, and `Cooling_Fluid_Temperature`.

### Polling Loops and Latency Constraints
Because frequency regulation requires responses within 500 milliseconds, the edge daemon runs an event-driven control loop:
1. Deterministic 100ms timer ticks initiate non-blocking async Modbus poll frames.
2. Read values are checked against safety threshold envelopes (e.g. cell temperature $> 55^\circ\text{C}$ or voltage imbalance $> 50\text{mV}$).
3. State-machine transitions occur locally if grid loss is detected (islanding mode transition).

## 3. High-Throughput Cloud Telemetry Pipeline
Edge controllers aggregate and compress high-frequency time-series telemetry before transmitting upstream:
- Telemetry batches are serialized into compact binary formats (Protobuf or lightweight JSON).
- Data is published via **MQTT (QoS 1)** over TLS 1.3 to AWS IoT Core or local broker clusters.
- On-device SQLite ring buffers persist up to 14 days of telemetry during internet backhaul outages, with automatic deduplicated catch-up replay upon reconnection.
