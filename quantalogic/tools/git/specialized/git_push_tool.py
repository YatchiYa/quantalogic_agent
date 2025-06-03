"""Tool for pushing changes to a remote Git repository."""

import os
from pathlib import Path
from typing import Optional, Tuple, Literal
import re

from git import Repo, GitCommandError as GitPythonError
from loguru import logger
from pydantic import Field

from quantalogic.tools.tool import Tool, ToolArgument

# Base directory for all cloned repositories
REPOS_BASE_DIR = "/tmp"
REPOS_SUBDIR = "git_repos"

# Repository types
REPO_TYPE_GITHUB = "github"
REPO_TYPE_GITLAB = "gitlab"
REPO_TYPE_UNKNOWN = "unknown"

class GitPushTool(Tool):
    """Tool for pushing changes to a remote Git repository."""

    name: str = "git_push_tool"
    description: str = (
        "Pushes changes to a remote Git repository with the following capabilities:\n"
        "- Pushes commits to the remote repository\n"
        "- Can auto-commit any pending changes if needed\n"
        "- Handles authentication for GitHub and GitLab repositories\n"
        "Repositories are organized in agent-specific directories under /tmp/agent_id/git_repos/ where an agent_id is always provided during tool initialization.\n"
        "This ensures isolation between different agents working with the same repositories."
    )
    need_validation: bool = False
    auth_token: str = Field(default=None, description="Authentication token for private repositories (GitHub or GitLab)")
    agent_id: Optional[str] = Field(default=None, description="Agent ID for directory organization")

    def __init__(self, auth_token: str = None, agent_id: str = None, **data):
        """Initialize the tool with optional parameters.
        
        Args:
            auth_token: Authentication token for private repositories (GitHub or GitLab)
            agent_id: Agent ID for directory organization
            **data: Additional tool configuration data
        """
        super().__init__(**data)
        self.auth_token = auth_token
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
            description="The name of the branch to push",
            required=True,
            example="feature/new-feature",
        ),
        ToolArgument(
            name="auto_commit",
            arg_type="boolean",
            description="Whether to automatically commit any pending changes before pushing",
            required=False,
            default="True",
        ),
        ToolArgument(
            name="commit_message",
            arg_type="string",
            description="The commit message to use if auto-committing (defaults to 'Auto-commit before push')",
            required=False,
            default=None,
        ),
    ]

    def detect_repo_type(self, repo: Repo) -> Tuple[Literal["github", "gitlab", "unknown"], str, str]:
        """Detect the repository type and extract owner and repo name from a local repository.
        
        Args:
            repo: A Git repository object
            
        Returns:
            Tuple containing:
                - Repository type (github, gitlab, or unknown)
                - Owner/username
                - Repository name
        """
        try:
            # Get the remote URL
            remote_url = repo.remotes.origin.url
            
            # Clean the URL (remove auth token if present)
            # Handle both standard token and oauth2 token formats
            url = re.sub(r"https://[^@]+@", "https://", remote_url)
            url = re.sub(r"https://oauth2:[^@]+@", "https://", url)
            url = url.rstrip("/").rstrip(".git")
            
            logger.info(f"Detecting repo type from cleaned URL: {url}")
            
            # GitHub pattern
            github_pattern = r"https://github\.com/([^/]+)/([^/]+)"
            github_match = re.match(github_pattern, url)
            if github_match:
                return REPO_TYPE_GITHUB, github_match.group(1), github_match.group(2)
            
            # GitLab pattern
            gitlab_pattern = r"https://gitlab\.com/([^/]+)/([^/]+)"
            gitlab_match = re.match(gitlab_pattern, url)
            if gitlab_match:
                return REPO_TYPE_GITLAB, gitlab_match.group(1), gitlab_match.group(2)
            
            logger.warning(f"Unknown repository type for URL: {url}")
            return REPO_TYPE_UNKNOWN, "", ""
            
        except Exception as e:
            logger.error(f"Error detecting repository type: {str(e)}")
            return REPO_TYPE_UNKNOWN, "", ""

    def push_changes(self, repo: Repo, branch_name: str, auto_commit: bool = True, commit_message: str = None) -> str:
        """Push changes to the remote repository. Can automatically commit changes if needed.
        
        Args:
            repo: The Git repository object
            branch_name: Name of the branch to push
            auto_commit: Whether to automatically commit any pending changes before pushing
            commit_message: Commit message to use if auto-committing (defaults to "Auto-commit before push")
            
        Returns:
            str: Success message
            
        Raises:
            ValueError: If push fails
        """
        try:
            # Ensure we're on the correct branch
            current_branch = repo.active_branch.name
            if current_branch != branch_name:
                logger.warning(f"Not on the specified branch. Current branch: {current_branch}, Requested branch: {branch_name}")
                return f"Error: Not on the specified branch. Current branch: {current_branch}, Requested branch: {branch_name}. Please checkout the correct branch first."
            
            # Check if there are uncommitted changes that need to be committed first
            has_changes = False
            commit_result = ""
            
            if auto_commit:
                # Check for uncommitted changes
                status = repo.git.status(porcelain=True)
                has_changes = bool(status.strip())
                
                if has_changes:
                    logger.info("Detected uncommitted changes, auto-committing before push")
                    
                    # Use default commit message if none provided
                    if not commit_message:
                        commit_message = "Auto-commit before push"
                    
                    # Add all changes to staging
                    logger.info("Adding all changes to staging")
                    repo.git.add(A=True)
                    
                    # Commit the changes
                    logger.info(f"Committing with message: {commit_message}")
                    commit = repo.index.commit(commit_message)
                    
                    if commit:
                        logger.info(f"Auto-committed changes with hash: {commit.hexsha}")
                        commit_result = f"Successfully committed changes with hash: {commit.hexsha[:8]}"
                    else:
                        logger.warning("Auto-commit did not create a commit object")
                        return "Error: Failed to auto-commit changes before push"
            
            # Detect repository type
            repo_type, owner, repo_name = self.detect_repo_type(repo)
            
            # Get the remote URL
            remote_url = repo.remotes.origin.url
            original_url = remote_url
            
            # Check if authentication is already in the URL
            auth_already_present = "@" in remote_url and "://" in remote_url
            
            # Add authentication token if needed and not already present
            if self.auth_token and "https://" in remote_url and not auth_already_present:
                # First, clean the URL to ensure we don't have any existing auth
                clean_url = re.sub(r"https://[^@]+@", "https://", remote_url)
                
                if repo_type == REPO_TYPE_GITHUB:
                    # GitHub uses token@ format
                    auth_url = clean_url.replace("https://", f"https://{self.auth_token}@")
                elif repo_type == REPO_TYPE_GITLAB:
                    # GitLab uses oauth2: format
                    auth_url = clean_url.replace("https://", f"https://oauth2:{self.auth_token}@")
                else:
                    # Generic approach for unknown types
                    auth_url = clean_url.replace("https://", f"https://{self.auth_token}@")
                
                # Set the new remote URL with authentication
                logger.info(f"Setting authenticated URL for {repo_type}")
                repo.remotes.origin.set_url(auth_url)
                remote_url = auth_url
            
            try:
                # Push changes
                logger.info(f"Pushing to branch: {branch_name}")
                push_info = repo.remotes.origin.push(refspec=f"refs/heads/{branch_name}:refs/heads/{branch_name}")
                
                # Check push results
                for info in push_info:
                    if info.flags & info.ERROR:
                        error_msg = f"Failed to push to branch '{branch_name}': {info.summary}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                
                # Prepare result message including auto-commit info if it happened
                if auto_commit and has_changes and commit_result:
                    logger.info(f"Successfully pushed changes to branch: {branch_name} (with auto-commit)")
                    return f"Successfully auto-committed and pushed changes to branch: {branch_name}"
                else:
                    logger.info(f"Successfully pushed changes to branch: {branch_name}")
                    return f"Successfully pushed changes to branch: {branch_name}"
            finally:
                # Reset the remote URL to the original one
                if remote_url != original_url:
                    logger.info("Resetting remote URL to original")
                    repo.remotes.origin.set_url(original_url)
            
        except Exception as e:
            error_msg = str(e)
            # Remove sensitive information from error message if present
            if self.auth_token:
                error_msg = error_msg.replace(self.auth_token, "***")
            logger.error(f"Failed to push changes: {error_msg}")
            raise ValueError(f"Failed to push changes: {error_msg}")

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
        auto_commit: bool = True,
        commit_message: str = None,
        agent_id: str = None,
    ) -> str:
        """Executes the push operation on the specified repository.

        Args:
            repo_path: Local path to the Git repository
            branch_name: Name of the branch to push
            auto_commit: Whether to automatically commit any pending changes before pushing
            commit_message: The commit message to use if auto-committing
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
            
            # Execute the push operation
            return self.push_changes(repo, branch_name, auto_commit, commit_message)
            
        except GitPythonError as e:
            error_msg = str(e)
            # Remove sensitive information from error message if present
            if self.auth_token:
                error_msg = error_msg.replace(self.auth_token, "***")
            logger.error(f"Git error: {error_msg}")
            raise ValueError(f"Git error: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            # Remove sensitive information from error message if present
            if self.auth_token and isinstance(error_msg, str):
                error_msg = error_msg.replace(self.auth_token, "***")
            logger.error(f"An error occurred: {error_msg}")
            raise ValueError(f"An error occurred: {error_msg}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Default values
    repo_path = "/tmp/agent_id/git_repos/my_repo"
    branch_name = "feature/new-feature"
    auto_commit = True
    commit_message = "Auto-commit before push"
    token = None
    agent_id = "test_agent"
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    if len(sys.argv) > 2:
        branch_name = sys.argv[2]
    if len(sys.argv) > 3:
        commit_message = sys.argv[3]
    if len(sys.argv) > 4:
        token = sys.argv[4]
    if len(sys.argv) > 5:
        agent_id = sys.argv[5]
    
    # Create and execute the tool
    tool = GitPushTool(auth_token=token, agent_id=agent_id)
    result = tool.execute(repo_path=repo_path, branch_name=branch_name, auto_commit=auto_commit, commit_message=commit_message)
    print(result)
