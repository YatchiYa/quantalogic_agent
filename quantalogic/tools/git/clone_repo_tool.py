"""Tool for cloning Git repositories with support for both public and private repositories from GitHub and GitLab."""

import os
import shutil
from pathlib import Path
from typing import Tuple, Literal, Optional, Dict, Any, Union
import re

import requests
from git import Repo
from git.exc import GitCommandError
from loguru import logger
from pydantic import Field

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.list_directory_tool import ListDirectoryTool
from quantalogic.tools.git.specialized.git_list_branches_tool import GitListBranchesTool

# Base directory for all cloned repositories
REPOS_BASE_DIR = "/tmp"
REPOS_SUBDIR = "git_repos"

# Repository types
REPO_TYPE_GITHUB = "github"
REPO_TYPE_GITLAB = "gitlab"
REPO_TYPE_UNKNOWN = "unknown"

class CloneRepoTool(Tool):
    """Tool for cloning Git repositories from GitHub or GitLab."""

    name: str = "clone_repo_tool"
    description: str = (
        "Clones a Git repository (public or private) from GitHub or GitLab to a specified location. "
        "Always clones the default branch of the repository."
        "Automatically handles authentication for private repositories using the provided token."
        "Automatically creates a new branch and checkout on it, with a better name."
        "Repositories are cloned into agent-specific directories under /tmp/agent_id/git_repos/."
    )
    need_validation: bool = False
    auth_token: str = Field(default=None, description="Authentication token for private repositories (GitHub or GitLab)")
    default_repo_url: str = Field(default="", description="Default repository URL to use if none is provided during execution")
    default_create_branch: str | None = Field(default=None, description="Default branch to create and checkout after cloning")
    agent_id: Optional[str] = None

    def __init__(self, auth_token: str = None, default_repo_url: str = "", default_create_branch: str = None, agent_id: str = None, **data):
        """Initialize the tool with optional parameters.
        
        Args:
            auth_token: Authentication token for private repositories (GitHub or GitLab)
            default_repo_url: Default repository URL to use if none is provided during execution
            default_create_branch: Default branch to create and checkout after cloning
            agent_id: Optional agent ID for directory organization
            **data: Additional tool configuration data
        """
        super().__init__(**data)
        self.auth_token = auth_token
        self.default_repo_url = default_repo_url
        self.default_create_branch = default_create_branch
        self.agent_id = agent_id

    arguments: list = [
        ToolArgument(
            name="repo_url",
            arg_type="string",
            description="The URL of the Git repository to clone (HTTPS format)",
            required=False,  # Changed to False since we can use default_repo_url
            example="https://github.com/username/repo.git or https://gitlab.com/username/repo.git",
        ),
        ToolArgument(
            name="target_path",
            arg_type="string",
            description=f"The local path where the repository should be cloned (will be adjusted to be within /tmp/agent_id/git_repos/)",
            required=True,
            example=f"/tmp/agent_id/git_repos/repo_name",
        ),
        ToolArgument(
            name="create_branch",
            arg_type="string",
            description="Name of a new branch to create and checkout after cloning (if provided)",
            required=False,  # Changed to False since we can use default_create_branch
            default=None,  # Will use self.default_create_branch if None
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
        
        Ensures the target directory is within /tmp/agent_id/git_repos/ and prepares it for cloning.
        If agent_id is provided, the repository will be cloned into a subdirectory for that agent.
        
        Args:
            target_path: Path where the repository will be cloned
            
        Raises:
            ValueError: If the target path is not within the appropriate directory
        """
        # Determine base directory based on agent_id
        if self.agent_id and str(self.agent_id).strip():
            agent_base_dir = os.path.join(REPOS_BASE_DIR, self.agent_id, REPOS_SUBDIR)
            logger.info(f"Using agent-specific directory: {agent_base_dir}")
            # Ensure agent-specific base directory exists
            os.makedirs(agent_base_dir, exist_ok=True)
        else:
            # Ensure base directory exists (fallback to /tmp/git_repos if no agent_id)
            fallback_dir = os.path.join(REPOS_BASE_DIR, REPOS_SUBDIR)
            os.makedirs(fallback_dir, exist_ok=True)
        
        # Convert to absolute path and ensure it's within the appropriate base directory
        abs_target = os.path.abspath(target_path)
        
        # If agent_id is provided, adjust the target path to be within the agent's directory
        if self.agent_id and str(self.agent_id).strip():
            agent_base_dir = os.path.join(REPOS_BASE_DIR, self.agent_id, REPOS_SUBDIR)
            
            # If path is not already in the agent's directory, adjust it
            if not abs_target.startswith(agent_base_dir):
                # If it's in REPOS_BASE_DIR but not in agent dir, move it to agent dir
                if abs_target.startswith(REPOS_BASE_DIR):
                    # Extract the part after REPOS_BASE_DIR
                    relative_path = os.path.relpath(abs_target, REPOS_BASE_DIR)
                    abs_target = os.path.join(agent_base_dir, relative_path)
                    target_path = abs_target
                    logger.info(f"Adjusted target path to agent directory: {abs_target}")
                else:
                    # If it's outside REPOS_BASE_DIR entirely, use the basename in agent dir
                    new_target = os.path.join(agent_base_dir, os.path.basename(abs_target))
                    abs_target = new_target
                    target_path = abs_target
                    logger.info(f"Moved target path to agent directory: {abs_target}")
        else:
            # Without agent_id, just ensure it's within fallback directory
            fallback_dir = os.path.join(REPOS_BASE_DIR, REPOS_SUBDIR)
            if not abs_target.startswith(fallback_dir):
                new_target = os.path.join(fallback_dir, os.path.basename(abs_target))
                abs_target = new_target
                target_path = abs_target
                logger.info(f"Adjusted target path to base directory: {abs_target}")
        
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

    def execute(self, target_path: str, repo_url: str = None, create_branch: str = None, agent_id: str = None) -> Union[str, Dict[str, Any]]:
        """Clones a Git repository to the specified path within REPOS_BASE_DIR.

        Args:
            repo_url: URL of the Git repository (GitHub or GitLab)
            target_path: Local path where to clone the repository (will be adjusted to be within /tmp/agent_id/git_repos/)
            create_branch: Name of a new branch to create and checkout after cloning
            agent_id: Optional agent ID to use for directory organization

        Returns:
            str: Path where the repository was cloned

        Raises:
            GitCommandError: If there's an error during cloning
            ValueError: If the parameters are invalid or target_path cannot be adjusted to the proper directory
        """
        try:
            # Update agent_id if provided in this call
            if agent_id is not None:
                logger.info(f"Setting agent_id from parameter: {agent_id}")
                self.agent_id = agent_id
                
            # Use default values if parameters are not provided
            actual_repo_url = repo_url if repo_url is not None else self.default_repo_url
            actual_create_branch = create_branch if create_branch is not None else self.default_create_branch
            
            # Validate that we have a repository URL
            if not actual_repo_url:
                raise ValueError("Repository URL must be provided either during initialization or execution")
            
            # Detect repository type
            repo_type, owner, repo = self.detect_repo_type(actual_repo_url)
            
            if repo_type == REPO_TYPE_UNKNOWN:
                logger.warning(f"Unknown repository type, will attempt to clone directly: {actual_repo_url}")
            
            # Ensure target_path is within the appropriate directory based on agent_id
            if self.agent_id and str(self.agent_id).strip():
                agent_base_dir = os.path.join(REPOS_BASE_DIR, self.agent_id, REPOS_SUBDIR)
                if not os.path.abspath(target_path).startswith(agent_base_dir):
                    # If it's in REPOS_BASE_DIR but not in agent dir
                    if os.path.abspath(target_path).startswith(REPOS_BASE_DIR):
                        rel_path = os.path.relpath(os.path.abspath(target_path), REPOS_BASE_DIR)
                        target_path = os.path.join(agent_base_dir, rel_path)
                    else:
                        target_path = os.path.join(agent_base_dir, os.path.basename(target_path))
                    logger.info(f"Adjusting target path to agent directory: {target_path}")
            else:
                # Without agent_id, ensure it's within fallback directory
                fallback_dir = os.path.join(REPOS_BASE_DIR, REPOS_SUBDIR)
                if not os.path.abspath(target_path).startswith(fallback_dir):
                    target_path = os.path.join(fallback_dir, os.path.basename(target_path))
                    logger.info(f"Adjusting target path to base directory: {target_path}")

            # Prepare target directory (remove if exists and create new)
            self._prepare_target_directory(target_path)

            # Check if repo is private and token is needed
            is_private = self.is_private_repo(actual_repo_url)
            
            if is_private and not self.auth_token:
                raise ValueError("Authentication token required for private repository")
            
            # Prepare the clone URL with auth token if needed
            clone_url = actual_repo_url
            
            # Always use token for GitHub repos in Docker environments when available
            # This prevents the 'could not read Username' error in non-interactive environments
            if self.auth_token:
                if repo_type == REPO_TYPE_GITHUB:
                    # GitHub uses token@ format
                    clone_url = actual_repo_url.replace("https://", f"https://{self.auth_token}@")
                    logger.info("Using token authentication for GitHub repository")
                elif repo_type == REPO_TYPE_GITLAB:
                    # GitLab uses oauth2: format
                    clone_url = actual_repo_url.replace("https://", f"https://oauth2:{self.auth_token}@")
                    logger.info("Using token authentication for GitLab repository")
                else:
                    # Generic approach for unknown types
                    clone_url = actual_repo_url.replace("https://", f"https://{self.auth_token}@")
                    logger.info("Using token authentication for repository")

            logger.info(f"Cloning {repo_type} repository to {target_path}")
            
            # Clone the repository (always clone default branch)
            try:
                logger.info("Cloning default branch of the repository")
                repo = Repo.clone_from(
                    url=clone_url,
                    to_path=target_path,
                )
            except GitCommandError as e:
                raise
            
            # Create and checkout a new branch if requested
            if actual_create_branch:
                try:
                    # Create a new branch
                    git = repo.git
                    git.checkout('-b', actual_create_branch)
                    logger.info(f"Created and checked out new branch: {actual_create_branch}")
                except GitCommandError as e:
                    logger.warning(f"Failed to create branch {actual_create_branch}: {str(e)}")
                    # Continue execution as the clone was successful

            logger.info(f"Successfully cloned repository to {target_path}")
            
            # List files in the cloned repository
            try:
                list_dir_tool = ListDirectoryTool(agent_id=self.agent_id)
                directory_listing = list_dir_tool.execute(
                    directory_path=target_path,
                    recursive="true",
                    max_depth="10",
                    start_line="1",
                    end_line="200",
                    agent_id=self.agent_id
                )
                
                # Build success message
                success_message = f"Repository successfully cloned to: {target_path}"
                if actual_create_branch:
                    success_message += f" and branch '{actual_create_branch}' was created and checked out"
                
                # Add directory listing to the success message
                if isinstance(directory_listing, dict) and directory_listing.get("status") == "success":
                    success_message += f"\n\nRepository contents:\n{directory_listing.get('answer')}"
                elif isinstance(directory_listing, str):
                    success_message += f"\n\nRepository contents:\n{directory_listing}"
                else:
                    success_message += "\n\nUnable to list repository contents."

                # List local branches
                try:
                    list_branches_tool = GitListBranchesTool(agent_id=self.agent_id)
                    local_branches_listing = list_branches_tool.execute(
                        repo_path=target_path,
                        list_type="local",
                        agent_id=self.agent_id
                    )
                    success_message += f"\n\n{local_branches_listing}"
                except Exception as e:
                    logger.warning(f"Failed to list local branches: {str(e)}")
                    success_message += "\n\nUnable to list local branches."

                # List remote branches
                try:
                    # Re-instantiate or ensure state is clean if necessary, though for this tool it might be fine
                    list_branches_tool_remote = GitListBranchesTool(agent_id=self.agent_id) 
                    remote_branches_listing = list_branches_tool_remote.execute(
                        repo_path=target_path,
                        list_type="remote",
                        agent_id=self.agent_id
                    )
                    success_message += f"\n\n{remote_branches_listing}"
                except Exception as e:
                    logger.warning(f"Failed to list remote branches: {str(e)}")
                    success_message += "\n\nUnable to list remote branches."
                    
                return {
                    "status": "success",
                    "answer": success_message
                }
            except Exception as e:
                logger.warning(f"Failed to list repository contents or branches: {str(e)}")
                # Return basic success message if listing fails
                success_message = f"Repository successfully cloned to: {target_path}"
                if actual_create_branch:
                    success_message += f" and branch '{actual_create_branch}' was created and checked out"
                success_message += "\n\nFurther repository exploration (file listing, branches) failed."
                return {
                    "status": "success", # Still success for clone, but with a note
                    "answer": success_message
                }

        except GitCommandError as e:
            error_msg = str(e)
            # Remove sensitive information from error message if present
            if self.auth_token:
                error_msg = error_msg.replace(self.auth_token, "***")
            logger.error(f"Failed to clone repository: {error_msg}")
            # raise GitCommandError(f"Failed to clone repository: {error_msg}", e.status)
            return {
                "status": "error",
                "answer": f"Failed to clone repository: {error_msg}"
            }
        
        except Exception as e:
            logger.error(f"An error occurred while cloning the repository: {str(e)}")
            # raise ValueError(f"An error occurred while cloning the repository: {str(e)}")
            return {
                "status": "error",
                "answer": f"An error occurred while cloning the repository: {str(e)}"
            }


if __name__ == "__main__":
    # Example usage for GitLab repository
    import sys
    
    # Default values
    repo_url ="https://github.com/novagen-conseil/selqgit.git" # "https://gitlab.com/quantalogic/ql_demo_private"
    token = "ghp.."
    target_path = f"{REPOS_BASE_DIR}/sqlgit"
    create_branch = "test"
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        repo_url = sys.argv[1]
    if len(sys.argv) > 2:
        token = sys.argv[2]
    if len(sys.argv) > 3:
        target_path = sys.argv[3]
    if len(sys.argv) > 4:
        create_branch = sys.argv[4]
    
    print("Example 1: Using default values provided during initialization")
    # Initialize the tool with default values
    tool1 = CloneRepoTool(
        auth_token=token,
        default_repo_url=repo_url,
        default_create_branch=create_branch
    )
    
    try:
        # Execute with only target_path, using defaults for other parameters
        result = tool1.execute(target_path=target_path)
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\nExample 2: Traditional usage with parameters during execution")
    # Initialize the tool with just the token
    tool2 = CloneRepoTool(auth_token=token)
    
    try:
        # Execute with all parameters specified
        result = tool2.execute(
            repo_url=repo_url,
            target_path=f"{target_path}_example2",
            create_branch=create_branch
        )
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")
