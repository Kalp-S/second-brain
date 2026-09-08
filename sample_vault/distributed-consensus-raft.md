---
title: "Distributed Consensus: The Raft Protocol"
tags: ["distributed-systems", "consensus", "algorithms", "raft"]
---

# Distributed Consensus: The Raft Protocol

The Raft consensus algorithm is designed for managing a replicated log across a cluster of state machines. Unlike Multi-Paxos, which is notoriously difficult to understand and implement correctly, Raft decomposes consensus into three independent sub-problems: Leader Election, Log Replication, and Safety.

## 1. Node States and Term Lifecycles
In Raft, at any given time, each server is in one of three states:
- **Leader**: Handles all client requests, replicates log entries to followers, and sends periodic heartbeats (`AppendEntries` RPC with empty entries).
- **Follower**: Passive state; responds to incoming RPCs from leaders and candidates. If a follower receives no communication within an **election timeout** (randomized between 150ms - 300ms), it transitions to Candidate.
- **Candidate**: Increments the current term, votes for itself, and broadcasts `RequestVote` RPCs to all peers.

Time in Raft is divided into arbitrary **Terms**, numbered with monotonically increasing integers. Each term acts as a logical clock, allowing servers to detect obsolete information such as stale leaders.

## 2. Leader Election and Split-Brain Prevention
When a follower's election timeout expires:
1. It transitions to Candidate and increments its `currentTerm`.
2. It casts a vote for itself and sends `RequestVote` RPCs in parallel to all other cluster members.
3. A candidate wins an election if it receives votes from a **strict majority ($N/2 + 1$)** of servers in the cluster.
4. Once elected, it immediately establishes authority by broadcasting heartbeats to suppress new elections.

### Randomized Timeouts
To prevent **split votes** (where multiple candidates initiate elections simultaneously and split the votes evenly), Raft uses randomized election timeouts. Because timeouts are chosen randomly within a window (e.g., 150–300ms), one server usually times out before the others, triggers an election, and wins before other candidates wake up.

## 3. Log Replication and Commitment
Once a leader is established, it begins servicing client commands:
1. The leader appends the client command to its own log as a new uncommitted entry.
2. The leader issues `AppendEntries` RPCs in parallel to all followers.
3. When the entry has been safely replicated on a majority of the servers, the entry is considered **committed**.
4. The leader applies the committed entry to its state machine and returns the execution result to the client.
5. The leader keeps track of the highest index known to be committed in its `commitIndex` field and includes this value in subsequent `AppendEntries` RPCs so followers apply committed entries to their local state machines.

## 4. Raft Safety Invariants
Raft guarantees the following essential invariants:
- **Election Safety**: At most one leader can be elected in a given term.
- **Leader Append-Only**: A leader never overwrites or truncates its own log; it only appends new entries.
- **Log Matching Property**: If two logs contain an entry with the same index and term, they are identical up to that index.
- **Leader Completeness**: If a log entry is committed in a given term, that entry will be present in the logs of the leaders for all higher-numbered terms.
