"""Live tables: WebSocket lobbies, matches between two players (or a player and a bot),
escrowed buy-ins and atomic settlement.

    common.py    UserRef, TableConfig, errors
    protocol.py  parsing and validating client messages
    core.py      TableCore: the synchronous, deterministic match state machine
    ledger.py    escrow / settle / refund (MemoryLedger for tests, SqlLedger for production)
    table.py     Table: the async runtime (connections, timers, disconnects, settlement)
    lobby.py     Lobby: creates tables, finds them, enforces one table per user
"""
