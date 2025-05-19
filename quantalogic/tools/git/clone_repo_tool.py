"""Tool for cloning Git repositories with support for both public and private repositories from GitHub and GitLab."""

import os
import shutil
from pathlib import Path
from typing import Tuple, Literal
import re

import requests
from git import Repo
from git.exc import GitCommandError
from loguru import logger
from pydantic import Field

from quantalogic.tools.tool import Tool, ToolArgument

# Base directory for all cloned repositories
REPOS_BASE_DIR = "/tmp/git_repos"

# Repository types
REPO_TYPE_GITHUB = "github"
REPO_TYPE_GITLAB = "gitlab"
REPO_TYPE_UNKNOWN = "unknown"

class CloneRepoTool(Tool):
    """Tool for cloning Git repositories from GitHub or GitLab."""

    name: str = "clone_repo_tool"
    description: str = (
        "Clones a Git repository (public or private) from GitHub or GitLab to a specified location. "
        "Automatically handles authentication for private repositories using the provided token."
        "Automatically creates a new branch and checkout on it, with a better name."
    )
    need_validation: bool = False
    auth_token: str = Field(default=None, description="Authentication token for private repositories (GitHub or GitLab)")

    def __init__(self, auth_token: str = None, **data):
        """Initialize the tool with an optional auth token.
        
        Args:
            auth_token: Authentication token for private repositories (GitHub or GitLab)
            **data: Additional tool configuration data
        """
        super().__init__(**data)
        self.auth_token = auth_token

    arguments: list = [
        ToolArgument(
            name="repo_url",
            arg_type="string",
            description="The URL of the Git repository to clone (HTTPS format)",
            required=True,
            example="https://github.com/username/repo.git or https://gitlab.com/username/repo.git",
        ),
        ToolArgument(
            name="target_path",
            arg_type="string",
            description=f"The local path where the repository should be cloned (must be within {REPOS_BASE_DIR})",
            required=True,
            example=f"{REPOS_BASE_DIR}/repo_name",
        ),
        ToolArgument(
            name="branch",
            arg_type="string",
            description="Specific branch to clone (defaults to main/master)",
            required=False,
            default="main",
        ),
        ToolArgument(
            name="create_branch",
            arg_type="string",
            description="Name of a new branch to create and checkout after cloning (if provided)",
            required=True,
        ),
    ]

    def detect_repo_type(self, repo_url: str) -> Tuple[Literal["github", "gitlab", "unknown"], str, str]:
        """Detect the repository type and extract owner and repo name.
        
        Args:
            repo_url: Repository URL in HTTPS format
            
        Returns:
            Tuple containing:
                - Repository type (github, gitlab, or unknown)
                - Owner/username
                - Repository name
        """
        # Clean the URL
        url = repo_url.rstrip("/").rstrip(".git")
        
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
        
        logger.warning(f"Unknown repository type for URL: {repo_url}")
        return REPO_TYPE_UNKNOWN, "", ""

    def is_private_repo(self, repo_url: str) -> bool:
        """Check if a repository is private.
        
        Args:
            repo_url: Repository URL in format https://github.com/username/repo.git or https://gitlab.com/username/repo.git
        
        Returns:
            bool: True if repository is private, False otherwise
        """
        try:
            # Detect repository type and extract owner/repo
            repo_type, owner, repo = self.detect_repo_type(repo_url)
            
            if repo_type == REPO_TYPE_UNKNOWN:
                logger.warning(f"Unknown repository type, assuming private: {repo_url}")
                return True
            
            if repo_type == REPO_TYPE_GITHUB:
                # GitHub API check
                response = requests.get(f"https://api.github.com/repos/{owner}/{repo}")
                
                if response.status_code == 404 and self.auth_token:
                    # Try again with token
                    headers = {"Authorization": f"token {self.auth_token}"}
                    response = requests.get(
                        f"https://api.github.com/repos/{owner}/{repo}",
                        headers=headers
                    )
                    return response.status_code == 200  # If accessible with token, it's private
                
                return False  # Repository is public
                
            elif repo_type == REPO_TYPE_GITLAB:
                # GitLab API check
                response = requests.get(f"https://gitlab.com/api/v4/projects/{owner}%2F{repo}")
                
                if response.status_code == 404 and self.auth_token:
                    # Try again with token
                    headers = {"Authorization": f"Bearer {self.auth_token}"}
                    response = requests.get(
                        f"https://gitlab.com/api/v4/projects/{owner}%2F{repo}",
                        headers=headers
                    )
                    return response.status_code == 200  # If accessible with token, it's private
                
                return False  # Repository is public
            
        except Exception as e:
            logger.warning(f"Error checking repository visibility: {str(e)}")
            return True  # Assume private if can't determine
        
        return True  # Default to private for safety

    def _prepare_target_directory(self, target_path: str) -> None:
        """Prepare the target directory for cloning.
        
        Ensures the target directory is within REPOS_BASE_DIR and prepares it for cloning.
        
        Args:
            target_path: Path where the repository will be cloned
            
        Raises:
            ValueError: If the target path is not within REPOS_BASE_DIR
        """
        # Ensure base directory exists
        os.makedirs(REPOS_BASE_DIR, exist_ok=True)
        
        # Convert to absolute path and ensure it's within REPOS_BASE_DIR
        abs_target = os.path.abspath(target_path)
        if not abs_target.startswith(REPOS_BASE_DIR):
            raise ValueError(f"Target directory must be within {REPOS_BASE_DIR}")
        
        if os.path.exists(target_path):
            logger.info(f"Target directory exists, removing: {target_path}")
            try:
                # Remove directory and all its contents
                shutil.rmtree(target_path)
            except Exception as e:
                logger.error(f"Error removing existing directory: {str(e)}")
                raise ValueError(f"Failed to remove existing directory: {str(e)}")
        
        # Create new empty directory
        os.makedirs(target_path, exist_ok=True)
        logger.info(f"Created clean target directory: {target_path}")

    def execute(self, repo_url: str, target_path: str, branch: str = "main", create_branch: str = None) -> str:
        """Clones a Git repository to the specified path within REPOS_BASE_DIR.

        Args:
            repo_url: URL of the Git repository (GitHub or GitLab)
            target_path: Local path where to clone the repository (must be within REPOS_BASE_DIR)
            branch: Branch to clone (defaults to main)

        Returns:
            str: Path where the repository was cloned

        Raises:
            GitCommandError: If there's an error during cloning
            ValueError: If the parameters are invalid or target_path is outside REPOS_BASE_DIR
        """
        try:
            # Detect repository type
            repo_type, owner, repo = self.detect_repo_type(repo_url)
            
            if repo_type == REPO_TYPE_UNKNOWN:
                logger.warning(f"Unknown repository type, will attempt to clone directly: {repo_url}")
            
            # Ensure target_path is within REPOS_BASE_DIR
            if not os.path.abspath(target_path).startswith(REPOS_BASE_DIR):
                target_path = os.path.join(REPOS_BASE_DIR, os.path.basename(target_path))
                logger.info(f"Adjusting target path to: {target_path}")

            # Prepare target directory (remove if exists and create new)
            self._prepare_target_directory(target_path)

            # Check if repo is private and token is needed
            is_private = self.is_private_repo(repo_url)
            
            if is_private and not self.auth_token:
                raise ValueError("Authentication token required for private repository")
            
            # Prepare the clone URL with auth token if needed
            clone_url = repo_url
            if is_private and self.auth_token:
                if repo_type == REPO_TYPE_GITHUB:
                    # GitHub uses token@ format
                    clone_url = repo_url.replace("https://", f"https://{self.auth_token}@")
                elif repo_type == REPO_TYPE_GITLAB:
                    # GitLab uses oauth2: format
                    clone_url = repo_url.replace("https://", f"https://oauth2:{self.auth_token}@")
                else:
                    # Generic approach for unknown types
                    clone_url = repo_url.replace("https://", f"https://{self.auth_token}@")

            logger.info(f"Cloning {repo_type} repository to {target_path}")
            
            # Clone the repository
            repo = Repo.clone_from(
                url=clone_url,
                to_path=target_path,
                branch=branch,
            )
            
            # Create and checkout a new branch if requested
            if create_branch:
                try:
                    # Create a new branch
                    git = repo.git
                    git.checkout('-b', create_branch)
                    logger.info(f"Created and checked out new branch: {create_branch}")
                except GitCommandError as e:
                    logger.warning(f"Failed to create branch {create_branch}: {str(e)}")
                    # Continue execution as the clone was successful

            logger.info(f"Successfully cloned repository to {target_path}")
            success_message = f"Repository successfully cloned to: {target_path}"
            if create_branch:
                success_message += f" and branch '{create_branch}' was created and checked out"
            return success_message

        except GitCommandError as e:
            error_msg = str(e)
            # Remove sensitive information from error message if present
            if self.auth_token:
                error_msg = error_msg.replace(self.auth_token, "***")
            logger.error(f"Failed to clone repository: {error_msg}")
            raise GitCommandError(f"Failed to clone repository: {error_msg}", e.status)
        
        except Exception as e:
            logger.error(f"An error occurred while cloning the repository: {str(e)}")
            raise ValueError(f"An error occurred while cloning the repository: {str(e)}")


if __name__ == "__main__":
    # Example usage for GitLab repository
    import sys
    
    # Default values
    repo_url = "https://gitlab.com/quantalogic/ql_demo_private"
    token = "gl...."
    target_path = f"{REPOS_BASE_DIR}/cvezdez"
    branch = "main"
    create_branch = None
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        repo_url = sys.argv[1]
    if len(sys.argv) > 2:
        token = sys.argv[2]
    if len(sys.argv) > 3:
        target_path = sys.argv[3]
    if len(sys.argv) > 4:
        branch = sys.argv[4]
    if len(sys.argv) > 5:
        create_branch = sys.argv[5]
    
    # Initialize and run the tool
    tool = CloneRepoTool(auth_token=token)
    
    try:
        result = tool.execute(repo_url=repo_url, target_path=target_path, branch=branch, create_branch=create_branch)
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")
