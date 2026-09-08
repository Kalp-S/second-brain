---
title: "Database Isolation Levels, 2PL, and MVCC Internals"
tags: ["databases", "concurrency", "acid", "mvcc", "sql"]
---

# Database Isolation Levels, 2PL, and MVCC Internals

Understanding relational database isolation anomalies and concurrency control algorithms is fundamental to designing resilient distributed data architectures.

## 1. The ANSI SQL Isolation Levels & Anomalies
The ANSI SQL-92 standard defined four isolation levels in terms of phenomena they prohibit:
- **Read Uncommitted**: Suffers from Dirty Reads (reading uncommitted mutations from another transaction).
- **Read Committed**: Prohibits dirty reads; suffers from Non-Repeatable Reads (reading the same row twice within a transaction yields different values because an intermediate transaction committed an update).
- **Repeatable Read**: Guarantees that any row read once will maintain its value throughout the transaction; historically suffered from Phantom Reads (new rows inserted by another concurrent transaction matching a `WHERE` predicate).
- **Serializable**: Guarantees that concurrent transactions execute with equivalent outcome to some strictly serial execution order.

### Modern Phenomena Not Captured by ANSI SQL
Modern research (Berenson et al., 1995) identified anomalies that ANSI SQL failed to define:
- **Write Skew**: Occurs under Snapshot Isolation when two concurrent transactions read overlapping data sets, satisfy an invariant constraint locally, but make disjoint updates that collectively violate the global business invariant (e.g., maintaining at least one doctor on call).

## 2. Concurrency Control Mechanisms: 2PL vs MVCC
Databases enforce isolation through two contrasting philosophies:

### Two-Phase Locking (2PL) - Pessimistic
- **Growing Phase**: Locks are acquired (Shared locks for reads, Exclusive locks for writes); no locks can be released.
- **Shrinking Phase**: Locks are released; no new locks can be acquired.
- **Strict 2PL (SS2PL)**: Exclusive locks must be held until the transaction commits or aborts, preventing cascading aborts.
- **Trade-off**: High contention, frequent deadlocks, and readers block writers while writers block readers.

### Multi-Version Concurrency Control (MVCC) - Optimistic / Snapshot
- Readers never block writers, and writers never block readers.
- Every row write creates a new physical tuple version with transaction identifiers (`xmin` creation tx, `xmax` deletion tx).
- When a transaction begins, it receives a logical snapshot timestamp. It only sees row versions where `xmin < tx_id` and `xmax > tx_id` (or uncommitted).
- Old obsolete row versions are reclaimed asynchronously by a vacuum process (e.g. Postgres `VACUUM` worker).
