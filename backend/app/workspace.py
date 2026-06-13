"""
Workspace management for RFP isolation
"""

from datetime import datetime
from typing import Dict, Any, Optional
from langgraph.checkpoint.memory import MemorySaver

# Global workspace store
_workspaces: Dict[str, Dict[str, Any]] = {}
_memory_saver = MemorySaver()


class WorkspaceManager:
    """Manage separate workspaces for each RFP"""
    
    @staticmethod
    def create_workspace(rfp_id: str, rfp_text: str, metadata: Optional[Dict] = None) -> str:
        """Create a new workspace for an RFP"""
        
        if rfp_id in _workspaces:
            print(f"⚠️ Workspace {rfp_id} already exists, updating...")
        
        _workspaces[rfp_id] = {
            "rfp_id": rfp_id,
            "rfp_text": rfp_text[:1000],  # Store preview only
            "metadata": metadata or {},
            "state": None,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "updated_at": None
        }
        
        print(f"✓ Workspace created: {rfp_id}")
        return rfp_id
    
    @staticmethod
    def get_workspace(rfp_id: str) -> Optional[Dict]:
        """Get workspace by ID"""
        return _workspaces.get(rfp_id)
    
    @staticmethod
    def update_workspace(rfp_id: str, state: Dict) -> None:
        """Update workspace state"""
        if rfp_id in _workspaces:
            _workspaces[rfp_id]["state"] = state
            _workspaces[rfp_id]["status"] = "completed"
            _workspaces[rfp_id]["updated_at"] = datetime.now().isoformat()
            print(f"✓ Workspace updated: {rfp_id}")
    
    @staticmethod
    def delete_workspace(rfp_id: str) -> bool:
        """Delete a workspace"""
        if rfp_id in _workspaces:
            del _workspaces[rfp_id]
            print(f"✓ Workspace deleted: {rfp_id}")
            return True
        return False
    
    @staticmethod
    def list_workspaces() -> list:
        """List all workspaces"""
        return [
            {
                "rfp_id": w["rfp_id"],
                "status": w["status"],
                "created_at": w["created_at"],
                "updated_at": w["updated_at"],
                "has_results": w["state"] is not None
            }
            for w in _workspaces.values()
        ]
    
    @staticmethod
    def get_thread_config(rfp_id: str) -> Dict:
        """Get thread config for LangGraph memory isolation"""
        return {"configurable": {"thread_id": f"workspace_{rfp_id}"}}
    
    @staticmethod
    def get_memory_saver() -> MemorySaver:
        """Get the memory saver for LangGraph"""
        return _memory_saver


# Convenience functions
def get_workspace_state(rfp_id: str) -> Optional[Dict]:
    """Retrieve state for a specific workspace"""
    workspace = WorkspaceManager.get_workspace(rfp_id)
    if workspace:
        return workspace.get("state")
    return None


def get_all_workspaces() -> list:
    """Get all workspaces"""
    return WorkspaceManager.list_workspaces()