"""Session-level chat history. Deliberately separate from the vector database."""
from __future__ import annotations

import threading
from collections import OrderedDict, deque

from backend.schemas.chat import Turn


class ConversationStore:
    def __init__(self, max_turns: int = 6, max_sessions: int = 500):
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, deque[Turn]] = OrderedDict()
        self._lock = threading.Lock()

    def history(self, session_id: str) -> list[Turn]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def add(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            dq = self._sessions.setdefault(session_id, deque(maxlen=self.max_turns))
            dq.append(turn)
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
