"""Global application state management."""
from .ServerState import ServerState
from .AgentState import AgentState

# Initialize global states
server_state = ServerState()
agent_state = AgentState()

__all__ = ['server_state', 'agent_state']
