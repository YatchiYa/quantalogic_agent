"""Tool for performing Git operations like branch creation, checkout, commit, and push with GitHub and GitLab support."""

import os
from pathlib import Path
from typing import Tuple, Literal, List, Optional, Dict
import re
import sys

import requests
from git import Repo, GitCommandError as GitPythonError
from loguru import logger
from pydantic import Field

from quantalogic.tools.tool import Tool, ToolArgument

# Repository types
REPO_TYPE_GITHUB = "github"
REPO_TYPE_GITLAB = "gitlab"
REPO_TYPE_UNKNOWN = "unknown"


class GitOperationsTool(Tool):
    """Tool for performing Git operations like branch creation, checkout, commit, and push."""

    name: str = "git_operations_tool"
    description: str = (
        "Performs Git operations on a local repository with the following capabilities:\n"
        "- list_branches: Shows all available local branches with the current branch marked\n"
        "- list_remote_branches: Shows all remote branches (equivalent to git branch -r)\n"
        "- create_branch: Creates a new branch from the current HEAD\n"
        "- checkout: Switches to an existing branch\n"
        "- commit: Automatically adds all changes (equivalent to 'git add .') and commits with the provided message\n"
        "- push: Pushes commits to the remote repository and auto-commits any pending changes if needed\n"
        "- pull: Fetches and merges changes from the remote repository\n\n"
        "- branch_name is required for create, checkout, commit to, or push to. (not required for 'list_branches' or 'list_remote_branches' operations)\n"
        "Automatically handles authentication for GitHub and GitLab repositories using the provided token."
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
            name="repo_path",
            arg_type="string",
            description="The local path to the Git repository (must be within /tmp/git_repos)",
            required=True,
            example="/tmp/git_repos/my_repo",
        ),
        ToolArgument(
            name="operation",
            arg_type="string",
            description="The Git operation to perform: 'create_branch', 'checkout', 'commit', 'push', 'pull', 'list_branches', or 'list_remote_branches'",
            required=True,
            example="create_branch",
        ),
        ToolArgument(
            name="branch_name",
            arg_type="string",
            description="The name of the branch to create, checkout, commit to, or push to (not required for 'list_branches' or 'list_remote_branches' operations)",
            required=False,
            default=None,
            example="feature/new-feature",
        ),
        ToolArgument(
            name="commit_message",
            arg_type="string",
            description="The commit message (required for 'commit' operation)",
            required=False,
            default=None,
        ),
        # Removed files argument as we now automatically add all files
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

    def create_branch(self, repo: Repo, branch_name: str) -> str:
        """Create a new branch in the repository.
        
        Args:
            repo: The Git repository object
            branch_name: Name of the branch to create
            
        Returns:
            str: Success message
            
        Raises:
            ValueError: If the branch already exists or creation fails
        """
        try:
            # Check if branch already exists
            if branch_name in [branch.name for branch in repo.branches]:
                logger.warning(f"Branch '{branch_name}' already exists")
                return f"Branch '{branch_name}' already exists"
            
            # Create new branch from current HEAD
            new_branch = repo.create_head(branch_name)
            logger.info(f"Created new branch: {branch_name}")
            return f"Successfully created branch: {branch_name}"
            
        except Exception as e:
            logger.error(f"Failed to create branch '{branch_name}': {str(e)}")
            raise ValueError(f"Failed to create branch '{branch_name}': {str(e)}")

    def list_branches(self, repo: Repo) -> str:
        """List all local branches in the repository.
        
        Args:
            repo: The Git repository object
            
        Returns:
            str: Formatted list of local branches with current branch marked
            
        Raises:
            ValueError: If listing branches fails
        """
        try:
            branches = [branch.name for branch in repo.branches]
            current_branch = repo.active_branch.name
            
            # Format the output with the current branch marked
            result = "Available branches:\n"
            for branch in sorted(branches):
                prefix = "* " if branch == current_branch else "  "
                result += f"{prefix}{branch}\n"
            
            logger.info(f"Listed {len(branches)} branches")
            return result
            
        except Exception as e:
            logger.error(f"Failed to list local branches: {str(e)}")
            raise ValueError(f"Failed to list local branches: {str(e)}")
    
    def list_remote_branches(self, repo: Repo) -> str:
        """List all remote branches in the repository (equivalent to git branch -r).
        
        Args:
            repo: The Git repository object
            
        Returns:
            str: Formatted list of remote branches
            
        Raises:
            ValueError: If listing remote branches fails
        """
        try:
            # Fetch from remote to ensure we have the latest remote branches
            logger.info("Fetching from remote to update remote branches information")
            repo.remotes.origin.fetch()
            
            # Get remote branches
            remote_branches = []
            for ref in repo.remote().refs:
                # Skip HEAD reference
                if ref.name.endswith('/HEAD'):
                    continue
                remote_branches.append(ref.name)
            
            # Format the output
            if not remote_branches:
                result = "No remote branches found"
            else:
                result = "Remote branches:\n"
                for branch in sorted(remote_branches):
                    result += f"  {branch}\n"
            
            logger.info(f"Listed {len(remote_branches)} remote branches")
            return result
            
        except Exception as e:
            logger.error(f"Failed to list remote branches: {str(e)}")
            raise ValueError(f"Failed to list remote branches: {str(e)}")
    
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

    def commit_changes(self, repo: Repo, commit_message: str, branch_name: str) -> str:
        """Add all changes and commit them in the repository.
        
        Args:
            repo: The Git repository object
            commit_message: Commit message
            branch_name: Name of the branch to commit to
            
        Returns:
            str: Success message with commit hash
            
        Raises:
            ValueError: If there are no changes to commit or commit fails
        """
        try:
            # Ensure we're on the correct branch
            current_branch = repo.active_branch.name
            if current_branch != branch_name:
                logger.warning(f"Not on the specified branch. Current branch: {current_branch}, Requested branch: {branch_name}")
                return f"Error: Not on the specified branch. Current branch: {current_branch}, Requested branch: {branch_name}. Please checkout the correct branch first."
            
            # Get status before adding files to track what will be added
            status_before = repo.git.status(porcelain=True)
            logger.info(f"Status before adding files:\n{status_before if status_before else 'Working directory clean'}")
            
            # Add all changes to staging (equivalent to 'git add .')
            logger.info("Adding all changes to staging")
            repo.git.add(A=True)
            
            # Verify files were added by checking status again
            status_after = repo.git.status(porcelain=True)
            logger.info(f"Status after adding files:\n{status_after if status_after else 'Working directory clean'}")
            
            # Check if there are changes to commit
            staged_files = [line for line in status_after.splitlines() if line.startswith(('M', 'A', 'D', 'R'))]
            if not staged_files and not repo.index.diff("HEAD"):
                logger.warning("No changes to commit")
                return "No changes to commit"
            
            # Log what's being committed
            logger.info(f"Committing {len(staged_files)} files/changes")
            
            # Commit changes
            commit = repo.index.commit(commit_message)
            
            # Verify the commit was successful
            if commit:
                logger.info(f"Committed changes with hash: {commit.hexsha}")
                
                # Get list of files that were committed
                committed_files = commit.stats.files
                file_list = "\n - " + "\n - ".join(committed_files.keys()) if committed_files else "No files changed"
                
                return f"Successfully committed changes with hash: {commit.hexsha[:8]} on branch: {branch_name}\nCommitted files: {file_list}"
            else:
                logger.error("Commit operation completed but no commit object was returned")
                raise ValueError("Commit operation failed to create a commit")
            
        except GitPythonError as e:
            logger.error(f"Git error during commit: {str(e)}")
            raise ValueError(f"Git error during commit: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to commit changes: {str(e)}")
            raise ValueError(f"Failed to commit changes: {str(e)}")

    def pull_changes(self, repo: Repo, branch_name: str) -> str:
        """Pull changes from the remote repository (fetch and merge).
        
        Args:
            repo: The Git repository object
            branch_name: Name of the branch to pull changes into
            
        Returns:
            str: Success message with details of what was pulled
            
        Raises:
            ValueError: If pull fails
        """
        try:
            # Ensure we're on the correct branch
            current_branch = repo.active_branch.name
            if current_branch != branch_name:
                logger.warning(f"Not on the specified branch. Current branch: {current_branch}, Requested branch: {branch_name}")
                return f"Error: Not on the specified branch. Current branch: {current_branch}, Requested branch: {branch_name}. Please checkout the correct branch first."
            
            # Detect repository type for authentication
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
                # Get the current commit hash before pulling
                before_pull_commit = repo.head.commit.hexsha
                logger.info(f"Current commit before pull: {before_pull_commit[:8]}")
                
                # Fetch changes from remote
                logger.info(f"Fetching changes from remote for branch: {branch_name}")
                fetch_info = repo.remotes.origin.fetch()
                logger.info(f"Fetch completed with {len(fetch_info)} refs updated")
                
                # Pull changes (fetch + merge)
                logger.info(f"Pulling changes into branch: {branch_name}")
                pull_info = repo.remotes.origin.pull()
                
                # Get the commit hash after pulling
                after_pull_commit = repo.head.commit.hexsha
                
                # Check if anything changed
                if before_pull_commit == after_pull_commit:
                    logger.info("Pull completed, but branch is already up to date")
                    return f"Branch '{branch_name}' is already up to date with remote"
                
                # Get details of what changed
                commit_range = f"{before_pull_commit[:8]}..{after_pull_commit[:8]}"
                changed_files = repo.git.diff("--name-only", commit_range).splitlines()
                
                # Format the output
                if changed_files:
                    file_list = "\n - " + "\n - ".join(changed_files) if changed_files else "No files changed"
                    logger.info(f"Pull completed successfully. {len(changed_files)} files changed")
                    return f"Successfully pulled changes into branch: {branch_name}\nFiles updated: {file_list}"
                else:
                    logger.info("Pull completed successfully, but no file changes detected")
                    return f"Successfully pulled changes into branch: {branch_name}, but no file changes detected"
                
            finally:
                # Reset the remote URL to the original one
                if remote_url != original_url:
                    logger.info("Resetting remote URL to original")
                    repo.remotes.origin.set_url(original_url)
            
        except GitPythonError as e:
            error_msg = str(e)
            # Remove sensitive information from error message if present
            if self.auth_token:
                error_msg = error_msg.replace(self.auth_token, "***")
            logger.error(f"Git error during pull: {error_msg}")
            raise ValueError(f"Failed to pull changes: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            # Remove sensitive information from error message if present
            if self.auth_token:
                error_msg = error_msg.replace(self.auth_token, "***")
            logger.error(f"Failed to pull changes: {error_msg}")
            raise ValueError(f"Failed to pull changes: {error_msg}")

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
            if auto_commit:
                # Check for uncommitted changes
                status = repo.git.status(porcelain=True)
                has_changes = bool(status.strip())
                
                if has_changes:
                    logger.info("Detected uncommitted changes, auto-committing before push")
                    
                    # Use default commit message if none provided
                    if not commit_message:
                        commit_message = "Auto-commit before push"
                    
                    # Commit the changes
                    commit_result = self.commit_changes(repo, commit_message, branch_name)
                    logger.info(f"Auto-commit result: {commit_result}")
                    
                    # If commit failed with an error (not just "No changes to commit"), propagate the error
                    if commit_result.startswith("Error:") and "No changes to commit" not in commit_result:
                        return commit_result
            
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
                if auto_commit and has_changes and "Successfully committed changes" in commit_result:
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

    def execute(
        self, 
        repo_path: str, 
        operation: str, 
        branch_name: str = None, 
        commit_message: str = None
    ) -> str:
        """Executes the requested Git operation on the specified repository.

        Args:
            repo_path: Local path to the Git repository (must be within /tmp/git_repos)
            operation: Git operation to perform ('create_branch', 'checkout', 'commit', 'push', 'list_branches', or 'list_remote_branches')
            branch_name: Name of the branch (required for all operations)
            commit_message: Commit message (required for 'commit' operation)

        Returns:
            str: Result message of the operation

        Raises:
            ValueError: If parameters are invalid or operation fails
            GitCommandError: If there's an error during Git operations
        """
        try:
            # Validate repo_path is within the allowed directory
            if not repo_path.startswith("/tmp/git_repos/"):
                raise ValueError(f"Repository path must be within /tmp/git_repos/: {repo_path}")
                
            # Validate repo_path exists
            if not os.path.exists(repo_path):
                raise ValueError(f"Repository path does not exist: {repo_path}")
            
            # Open the repository
            repo = Repo(repo_path)
            
            # Validate operation
            valid_operations = ["create_branch", "checkout", "commit", "push", "pull", "list_branches", "list_remote_branches"]
            if operation not in valid_operations:
                raise ValueError(f"Invalid operation: {operation}. Must be one of {valid_operations}")
            
            # Validate branch_name for operations that require it
            branch_required_operations = ["create_branch", "checkout", "commit", "push", "pull"]
            if operation in branch_required_operations:
                if not branch_name or branch_name.strip() == "":
                    raise ValueError(f"Branch name is required for '{operation}' operation")
            
            # Validate commit_message for commit operation
            if operation == "commit" and not commit_message:
                raise ValueError("Commit message is required for 'commit' operation")
            
            # Execute the requested operation
            if operation == "create_branch":
                return self.create_branch(repo, branch_name)
                
            elif operation == "checkout":
                return self.checkout_branch(repo, branch_name)
                
            elif operation == "commit":
                return self.commit_changes(repo, commit_message, branch_name)
                
            elif operation == "push":
                # Auto-commit before push if needed, using commit_message if provided
                return self.push_changes(repo, branch_name, auto_commit=True, commit_message=commit_message)
                
            elif operation == "pull":
                return self.pull_changes(repo, branch_name)
                
            elif operation == "list_branches":
                return self.list_branches(repo)
                
            elif operation == "list_remote_branches":
                return self.list_remote_branches(repo)
            
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
    repo_path = "/tmp/git_repos/my_repo"
    operation = "commit"  # Options: create_branch, checkout, commit, push, pull, list_branches, list_remote_branches
    branch_name = "feature/new-feature"
    commit_message = "Add new feature"
    # Files argument removed as we now automatically add all files
    token = None
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
        # Ensure repo_path is within allowed directory
        if not repo_path.startswith("/tmp/git_repos/"):
            print(f"Error: Repository path must be within /tmp/git_repos/: {repo_path}")
            sys.exit(1)
    if len(sys.argv) > 2:
        operation = sys.argv[2]
    if len(sys.argv) > 3:
        branch_name = sys.argv[3]
    if len(sys.argv) > 4:
        commit_message = sys.argv[4]
    if len(sys.argv) > 5:
        token = sys.argv[5]
    
    # Initialize and run the tool
    tool = GitOperationsTool(auth_token=token)
    
    try:
        result = tool.execute(
            repo_path=repo_path,
            operation=operation,
            branch_name=branch_name,
            commit_message=commit_message
        )
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")
