"""Tool for checking out Git branches in a repository."""

import os
from pathlib import Path
from typing import Optional
import re

from git import Repo, GitCommandError as GitPythonError
from loguru import logger
from pydantic import Field

from quantalogic.tools.tool import Tool, ToolArgument

# Base directory for all cloned repositories
REPOS_BASE_DIR = "/tmp"
REPOS_SUBDIR = "git_repos"

class GitCheckoutTool(Tool):
    """Tool for checking out branches in a Git repository."""

    name: str = "git_checkout_tool"
    description: str = (
        "Checks out a branch in a Git repository with the following capabilities:\n"
        "- Handles both local and remote branches\n"
        "- Automatically creates tracking branches when checking out remote branches\n"
        "- Mimics Git CLI behavior for branch checkout\n"
        "Repositories are organized in agent-specific directories under /tmp/agent_id/git_repos/ where an agent_id is always provided during tool initialization.\n"
        "This ensures isolation between different agents working with the same repositories."
    )
    need_validation: bool = False
    agent_id: Optional[str] = Field(default=None, description="Agent ID for directory organization")

    def __init__(self, agent_id: str = None, **data):
        """Initialize the tool with optional parameters.
        
        Args:
            agent_id: Agent ID for directory organization
            **data: Additional tool configuration data
        """
        super().__init__(**data)
        self.agent_id = agent_id

    arguments: list = [
        ToolArgument(
            name="repo_path",
            arg_type="string",
            description="The local path to the Git repository (must be within /tmp/agent_id/git_repos/)",
            required=True,
            example="/tmp/agent_id/git_repos/my_repo",
        ),
        ToolArgument(
            name="branch_name",
            arg_type="string",
            description="The name of the branch to checkout (can be local or remote)",
            required=True,
            example="feature/new-feature",
        ),
    ]

    def checkout_branch(self, repo: Repo, branch_name: str) -> str:
        """Checkout a branch in the repository. Handles both local and remote branches.
        Mimics Git CLI behavior by automatically creating tracking branches when needed.
        
        Args:
            repo: The Git repository object
            branch_name: Name of the branch to checkout
            
        Returns:
            str: Success message
            
        Raises:
            ValueError: If the branch doesn't exist or checkout fails
        """
        try:
            # First check if it's a local branch
            local_branches = [branch.name for branch in repo.branches]
            if branch_name in local_branches:
                # Simple case: local branch exists, just check it out
                branch = repo.branches[branch_name]
                branch.checkout()
                logger.info(f"Checked out local branch: {branch_name}")
                return f"Successfully checked out local branch: {branch_name}"
            
            # Check if it's a fully qualified remote branch (e.g., origin/branch-name)
            is_remote_branch = branch_name.startswith('origin/')
            if is_remote_branch:
                # Extract the branch name without the remote prefix
                local_branch_name = branch_name.split('/', 1)[1] if '/' in branch_name else branch_name
            else:
                # This is the key case for git checkout branch_name working in terminal
                # When branch_name is not a local branch but might be a remote branch without the origin/ prefix
                local_branch_name = branch_name
                # Check if there's a matching remote branch
                remote_branch_name = f"origin/{branch_name}"
            
            # Fetch to ensure we have the latest remote info
            logger.info("Fetching from remote to update branch information")
            repo.remotes.origin.fetch()
            
            # Get all remote refs
            remote_refs = {ref.name: ref for ref in repo.remote().refs}
            
            # Determine the remote branch name to use
            if is_remote_branch:
                remote_branch_name = branch_name
            else:
                remote_branch_name = f"origin/{branch_name}"
            
            # Check if the remote branch exists
            if remote_branch_name not in remote_refs:
                logger.warning(f"Remote branch '{remote_branch_name}' does not exist")
                raise ValueError(f"Branch '{branch_name}' does not exist locally or remotely. Please verify the branch name.")
            
            # Check if local branch with same name already exists
            if local_branch_name in local_branches:
                # If it exists, just checkout the local branch
                logger.info(f"Local branch '{local_branch_name}' already exists, checking it out")
                branch = repo.branches[local_branch_name]
                branch.checkout()
                return f"Checked out existing local branch: {local_branch_name}"
            
            # Create a new local branch that tracks the remote branch
            logger.info(f"Creating local tracking branch '{local_branch_name}' from '{remote_branch_name}'")
            
            # Create the tracking branch
            tracking_branch = repo.create_head(
                local_branch_name, 
                remote_refs[remote_branch_name]
            )
            tracking_branch.set_tracking_branch(remote_refs[remote_branch_name])
            
            # Checkout the new tracking branch
            tracking_branch.checkout()
            
            logger.info(f"Created and checked out tracking branch: {local_branch_name}")
            return f"Created and checked out local tracking branch '{local_branch_name}' from remote '{remote_branch_name}'"
            
        except Exception as e:
            logger.error(f"Failed to checkout branch '{branch_name}': {str(e)}")
            raise ValueError(f"Failed to checkout branch '{branch_name}': {str(e)}")

    def _validate_repo_path(self, repo_path: str, agent_id: str = None) -> str:
        """Validate and adjust repository path based on agent_id.
        
        Args:
            repo_path: Path to the Git repository
            
        Returns:
            str: Validated and potentially adjusted repository path
            
        Raises:
            ValueError: If the repository path is invalid or doesn't exist
        """
        # Determine base directory based on agent_id
        if agent_id and str(agent_id).strip():
            agent_base_dir = os.path.join(REPOS_BASE_DIR, agent_id, REPOS_SUBDIR)
            logger.info(f"Using agent-specific directory: {agent_base_dir}")
            
            # Convert to absolute path
            abs_path = os.path.abspath(repo_path)
            
            # Check if path is already in the agent's directory
            if not abs_path.startswith(agent_base_dir):
                # If it's in REPOS_BASE_DIR but not in agent dir, adjust it
                if abs_path.startswith(REPOS_BASE_DIR):
                    # Extract the part after REPOS_BASE_DIR
                    relative_path = os.path.relpath(abs_path, REPOS_BASE_DIR)
                    # If it contains git_repos but not in the agent's path
                    if REPOS_SUBDIR in relative_path:
                        parts = relative_path.split(REPOS_SUBDIR, 1)
                        if len(parts) > 1:
                            # Keep only the part after git_repos
                            new_path = os.path.join(agent_base_dir, parts[1].lstrip('/'))
                            logger.info(f"Adjusted repository path to agent directory: {new_path}")
                            return new_path
                    
                    # Otherwise use the full relative path
                    new_path = os.path.join(agent_base_dir, relative_path)
                    logger.info(f"Adjusted repository path to agent directory: {new_path}")
                    return new_path
                else:
                    # If outside REPOS_BASE_DIR entirely, use basename in agent dir
                    new_path = os.path.join(agent_base_dir, os.path.basename(abs_path))
                    logger.info(f"Moved repository path to agent directory: {new_path}")
                    return new_path
            return abs_path
        else:
            # Without agent_id, ensure it's within the fallback directory
            fallback_dir = os.path.join(REPOS_BASE_DIR, REPOS_SUBDIR)
            abs_path = os.path.abspath(repo_path)
            
            if not abs_path.startswith(fallback_dir):
                new_path = os.path.join(fallback_dir, os.path.basename(abs_path))
                logger.info(f"Adjusted repository path to base directory: {new_path}")
                return new_path
            return abs_path

    def execute(
        self, 
        repo_path: str, 
        branch_name: str,
        agent_id: str = None,
    ) -> str:
        """Executes the branch checkout operation on the specified repository.

        Args:
            repo_path: Local path to the Git repository
            branch_name: Name of the branch to checkout
            agent_id: Optional agent ID for directory organization

        Returns:
            str: Result message of the operation

        Raises:
            ValueError: If parameters are invalid or operation fails
            GitCommandError: If there's an error during Git operations
        """
        try:
            # Use instance agent_id if provided, otherwise use the parameter
            agent_id = agent_id or self.agent_id
            
            # Validate and adjust repo_path based on agent_id
            repo_path = self._validate_repo_path(repo_path, agent_id)
            
            # Validate repo_path exists
            if not os.path.exists(repo_path):
                raise ValueError(f"Repository path does not exist: {repo_path}")
            
            # Open the repository
            repo = Repo(repo_path)
            
            # Validate branch_name
            if not branch_name or branch_name.strip() == "":
                raise ValueError("Branch name is required")
            
            # Execute the branch checkout
            return self.checkout_branch(repo, branch_name)
            
        except GitPythonError as e:
            error_msg = str(e)
            logger.error(f"Git error: {error_msg}")
            raise ValueError(f"Git error: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"An error occurred: {error_msg}")
            raise ValueError(f"An error occurred: {error_msg}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Default values
    repo_path = "/tmp/agent_id/git_repos/my_repo"
    branch_name = "feature/new-feature"
    agent_id = "test_agent"
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    if len(sys.argv) > 2:
        branch_name = sys.argv[2]
    if len(sys.argv) > 3:
        agent_id = sys.argv[3]
    
    # Create and execute the tool
    tool = GitCheckoutTool(agent_id=agent_id)
    result = tool.execute(repo_path=repo_path, branch_name=branch_name)
    print(result)
