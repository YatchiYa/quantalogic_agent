"""
Git Tools Module

This module provides tools and utilities related to Git operations.
"""

from loguru import logger

# Explicit imports of all tools in the module
from .bitbucket_clone_repo_tool import BitbucketCloneTool
from .bitbucket_operations_tool import BitbucketOperationsTool
from .clone_repo_tool import CloneRepoTool
from .git_operations_tool import GitOperationsTool

from .specialized.git_create_branch_tool import GitCreateBranchTool
from .specialized.git_checkout_tool import GitCheckoutTool
from .specialized.git_commit_tool import GitCommitTool
from .specialized.git_push_tool import GitPushTool
from .specialized.git_pull_tool import GitPullTool
from .specialized.git_list_branches_tool import GitListBranchesTool


# Define __all__ to control what is imported with `from ... import *`
__all__ = [
    'BitbucketCloneTool',
    'BitbucketOperationsTool',
    'CloneRepoTool',
    'GitOperationsTool',
    'GitCreateBranchTool',
    'GitCheckoutTool',
    'GitCommitTool',
    'GitPushTool',
    'GitPullTool',
    'GitListBranchesTool',
]

# Optional: Add logging for import confirmation
logger.info("Git tools module initialized successfully.")
