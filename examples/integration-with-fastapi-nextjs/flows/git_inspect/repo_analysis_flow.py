"""Repository Analysis Flow - Analyzes Git repositories and provides comprehensive insights."""

import asyncio
import os
from typing import Any, Dict, List, Optional, Callable
import datetime
import json

import anyio
import typer
from loguru import logger
from pydantic import BaseModel, Field

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.git.clone_repo_tool import CloneRepoTool, REPOS_BASE_DIR
from quantalogic.tools.list_directory_tool import ListDirectoryTool
from quantalogic.tools.read_file_tool import ReadFileTool 
from ..service import event_observer

# Configure logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Define structured output models
class FileStatistics(BaseModel):
    """Statistics about files in the repository."""
    total_files: int
    file_types: Dict[str, int]
    largest_files: List[Dict[str, Any]]
    code_files_count: int
    documentation_files_count: int
    config_files_count: int
    
class DirectoryStructure(BaseModel):
    """Information about the repository directory structure."""
    top_level_dirs: List[str]
    depth: int
    most_files_dir: str
    
class CodeMetrics(BaseModel):
    """Code quality and quantity metrics."""
    language_breakdown: Dict[str, float]
    estimated_loc: int
    file_count_by_language: Dict[str, int]

class RepoAnalysis(BaseModel):
    """Complete repository analysis."""
    repo_name: str
    file_stats: FileStatistics
    directory_structure: DirectoryStructure
    code_metrics: CodeMetrics

class LLMRepoInsights(BaseModel):
    """LLM-generated insights about the repository."""
    architecture_assessment: str = Field(description="Assessment of the repository's architecture and organization")
    code_quality_insights: str = Field(description="Insights about code quality, patterns, and potential issues")
    main_technologies: str = Field(description="Main technologies, frameworks, and libraries used")
    documentation_assessment: str = Field(description="Assessment of documentation quality and completeness")
    improvement_suggestions: str = Field(description="Suggestions for improvements to the codebase")
    key_features: str = Field(description="Key features and functionality identified in the repository")
    final_assessment: str = Field(description="Overall assessment and summary of the repository")

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Helper function to get template paths
def get_template_path(template_name):
    """Get the full path to a template file."""
    template_dir = os.path.join(TEMPLATES_DIR, "repo_analysis")
    os.makedirs(template_dir, exist_ok=True)
    return os.path.join(template_dir, template_name)

# Create template files if they don't exist
def ensure_templates_exist():
    """Create template files if they don't exist."""
    templates = {
        "system_repo_analysis.j2": """You are a senior software architect and code quality expert. Your task is to analyze a repository and provide insightful feedback on its structure, code quality, and potential improvements.

Focus on providing actionable insights about:
1. The overall architecture and organization
2. Code quality patterns and potential issues
3. Main technologies, frameworks, and libraries used
4. Documentation quality and completeness
5. Suggestions for improvements
6. Key features and functionality
7. Overall assessment

Be specific, technical, and constructive in your analysis.""",
        
        "prompt_repo_analysis.j2": """# Repository Analysis Request

## Repository Information
Repository Name: {{ repo_name }}

## File Statistics
Total Files: {{ file_stats.total_files }}
Code Files: {{ file_stats.code_files_count }}
Documentation Files: {{ file_stats.documentation_files_count }}
Config Files: {{ file_stats.config_files_count }}

File Types:
{% for file_type, count in file_stats.file_types.items() %}
- {{ file_type }}: {{ count }}
{% endfor %}

## Directory Structure
Top-level directories:
{% for dir in directory_structure.top_level_dirs %}
- {{ dir }}
{% endfor %}

Directory depth: {{ directory_structure.depth }}
Directory with most files: {{ directory_structure.most_files_dir }}

## Code Metrics
Language breakdown:
{% for lang, percentage in code_metrics.language_breakdown.items() %}
- {{ lang }}: {{ percentage }}%
{% endfor %}

Estimated lines of code: {{ code_metrics.estimated_loc }}

## File Samples
{% if file_samples %}
Here are samples of key files from the repository:

{% for file in file_samples %}
### {{ file.path }}
```
{{ file.content }}
```
{% endfor %}
{% endif %}

Based on this information, please provide a comprehensive analysis of the repository.""",
    }
    
    for filename, content in templates.items():
        file_path = get_template_path(filename)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Created template file: {file_path}")

# Custom Observer for Workflow Events
async def repo_analysis_progress_observer(event: WorkflowEvent):
    """Observer for workflow events to provide progress updates."""
    if event.event_type == WorkflowEventType.WORKFLOW_STARTED:
        print(f"\n{'='*50}\n🔍 Starting Repository Analysis 🔍\n{'='*50}")
    elif event.event_type == WorkflowEventType.NODE_STARTED:
        print(f"\n🔄 [{event.node_name}] Starting...")
    elif event.event_type == WorkflowEventType.NODE_COMPLETED:
        if event.node_name == "clone_repository":
            print(f"✅ [{event.node_name}] Repository cloned successfully")
        elif event.node_name == "analyze_repository":
            print(f"✅ [{event.node_name}] Basic analysis completed")
        elif event.node_name == "llm_analysis":
            print(f"✅ [{event.node_name}] LLM analysis completed")
        else:
            print(f"✅ [{event.node_name}] Completed")
    elif event.event_type == WorkflowEventType.WORKFLOW_COMPLETED:
        print(f"\n{'='*50}\n🎉 Repository Analysis Finished 🎉\n{'='*50}")
    elif event.event_type == WorkflowEventType.TRANSITION_EVALUATED:
        logger.debug(f"Transition evaluated: {event.transition_from} -> {event.transition_to}")

# Workflow Nodes
@Nodes.define(output=None)
async def clone_repository(repo_url: str, auth_token: Optional[str] = None, branch: Optional[str] = None) -> dict:
    """Clone the repository to a local directory.
    
    Args:
        repo_url: URL of the Git repository
        auth_token: Authentication token for private repositories
        
    Returns:
        Dictionary with repository information
    """
    # Extract repo name from URL
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    target_path = os.path.join(REPOS_BASE_DIR, repo_name)
    
    # Clone the repository
    clone_tool = CloneRepoTool(auth_token="" if auth_token is None else auth_token)
    result = clone_tool.execute(repo_url=repo_url, target_path=target_path, branch=branch)
    
    logger.info(f"Repository cloned: {result}")
    return {
        "repo_name": repo_name,
        "repo_path": target_path,
        "clone_result": result
    }

@Nodes.define(output=None)
async def list_repository_contents(repo_path: str) -> dict:
    """List the contents of the repository.
    
    Args:
        repo_path: Path to the cloned repository
        
    Returns:
        Dictionary with repository contents
    """
    list_tool = ListDirectoryTool()
    result = list_tool.execute(
        directory_path=repo_path,
        recursive="true",
        max_depth="10",
        start_line="1",
        end_line="1000"
    )
    
    logger.info(f"Repository contents listed")
    return {
        "directory_listing": result
    }

def get_file_extension(file_path: str) -> str:
    """Get the file extension from a file path."""
    _, ext = os.path.splitext(file_path)
    return ext.lower()[1:] if ext else "no_extension"

def is_code_file(file_path: str) -> bool:
    """Check if a file is a code file based on its extension."""
    code_extensions = {
        'py', 'js', 'ts', 'java', 'c', 'cpp', 'h', 'hpp', 'cs', 'go', 'rb', 
        'php', 'scala', 'kt', 'rs', 'swift', 'sh', 'pl', 'pm', 'r', 'lua'
    }
    ext = get_file_extension(file_path)
    return ext in code_extensions

def is_documentation_file(file_path: str) -> bool:
    """Check if a file is a documentation file based on its extension or name."""
    doc_extensions = {'md', 'rst', 'txt', 'adoc', 'wiki'}
    doc_filenames = {'readme', 'contributing', 'license', 'changelog', 'authors', 'documentation'}
    
    ext = get_file_extension(file_path)
    filename = os.path.basename(file_path).lower()
    name_without_ext = os.path.splitext(filename)[0].lower()
    
    return ext in doc_extensions or name_without_ext in doc_filenames

def is_config_file(file_path: str) -> bool:
    """Check if a file is a configuration file based on its extension or name."""
    config_extensions = {'json', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf', 'xml'}
    config_filenames = {'config', 'settings', '.gitignore', '.env', 'dockerfile', 'makefile', 'setup.cfg'}
    
    ext = get_file_extension(file_path)
    filename = os.path.basename(file_path).lower()
    
    return ext in config_extensions or filename in config_filenames

def estimate_language_breakdown(file_types: Dict[str, int]) -> Dict[str, float]:
    """Estimate language breakdown based on file extensions."""
    code_extensions = {
        'py': 'Python',
        'js': 'JavaScript',
        'ts': 'TypeScript',
        'java': 'Java',
        'c': 'C',
        'cpp': 'C++',
        'h': 'C/C++ Header',
        'hpp': 'C++ Header',
        'cs': 'C#',
        'go': 'Go',
        'rb': 'Ruby',
        'php': 'PHP',
        'scala': 'Scala',
        'kt': 'Kotlin',
        'rs': 'Rust',
        'swift': 'Swift',
        'sh': 'Shell',
        'pl': 'Perl',
        'r': 'R',
        'lua': 'Lua'
    }
    
    language_counts = {}
    total_code_files = 0
    
    for ext, count in file_types.items():
        if ext in code_extensions:
            lang = code_extensions[ext]
            language_counts[lang] = language_counts.get(lang, 0) + count
            total_code_files += count
    
    # Convert counts to percentages
    if total_code_files > 0:
        language_breakdown = {lang: round((count / total_code_files) * 100, 1) 
                             for lang, count in language_counts.items()}
    else:
        language_breakdown = {}
    
    return language_breakdown

@Nodes.define(output="repo_analysis")
async def analyze_repository(repo_path: str, repo_name: str, directory_listing: str) -> RepoAnalysis:
    """Analyze the repository structure and content.
    
    Args:
        repo_path: Path to the cloned repository
        repo_name: Name of the repository
        directory_listing: Directory listing from list_repository_contents
        
    Returns:
        RepoAnalysis object with repository analysis
    """
    # Walk through the repository to collect file information
    file_types = {}
    largest_files = []
    total_files = 0
    code_files_count = 0
    documentation_files_count = 0
    config_files_count = 0
    dir_file_counts = {}
    
    for root, dirs, files in os.walk(repo_path):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
            
        # Count files in this directory
        rel_path = os.path.relpath(root, repo_path)
        if rel_path == '.':
            rel_path = ''
        dir_file_counts[rel_path] = len(files)
        
        for file in files:
            file_path = os.path.join(root, file)
            total_files += 1
            
            # Get file extension and update counts
            ext = get_file_extension(file_path)
            file_types[ext] = file_types.get(ext, 0) + 1
            
            # Check file type
            if is_code_file(file_path):
                code_files_count += 1
            elif is_documentation_file(file_path):
                documentation_files_count += 1
            elif is_config_file(file_path):
                config_files_count += 1
                
            # Track large files
            try:
                size = os.path.getsize(file_path)
                largest_files.append({
                    'path': os.path.relpath(file_path, repo_path),
                    'size': size
                })
            except:
                pass
    
    # Sort and limit largest files
    largest_files.sort(key=lambda x: x['size'], reverse=True)
    largest_files = largest_files[:5]
    
    # Get top-level directories
    top_level_dirs = [d for d in os.listdir(repo_path) 
                     if os.path.isdir(os.path.join(repo_path, d)) and d != '.git']
    
    # Find directory with most files
    most_files_dir = max(dir_file_counts.items(), key=lambda x: x[1], default=('', 0))[0] or 'root'
    
    # Calculate directory depth
    max_depth = 0
    for root, _, _ in os.walk(repo_path):
        rel_path = os.path.relpath(root, repo_path)
        if rel_path == '.':
            continue
        depth = len(rel_path.split(os.sep))
        max_depth = max(max_depth, depth)
    
    # Estimate language breakdown
    language_breakdown = estimate_language_breakdown(file_types)
    
    # Estimate lines of code (very rough estimate)
    estimated_loc = code_files_count * 200  # Assuming average of 200 lines per file
    
    # Create file count by language
    file_count_by_language = {}
    for ext, count in file_types.items():
        if ext in ['py', 'js', 'ts', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php']:
            language = {
                'py': 'Python',
                'js': 'JavaScript',
                'ts': 'TypeScript',
                'java': 'Java',
                'c': 'C',
                'cpp': 'C++',
                'cs': 'C#',
                'go': 'Go',
                'rb': 'Ruby',
                'php': 'PHP'
            }.get(ext, ext)
            file_count_by_language[language] = count
    
    # Create analysis object
    analysis = RepoAnalysis(
        repo_name=repo_name,
        file_stats=FileStatistics(
            total_files=total_files,
            file_types=file_types,
            largest_files=largest_files,
            code_files_count=code_files_count,
            documentation_files_count=documentation_files_count,
            config_files_count=config_files_count
        ),
        directory_structure=DirectoryStructure(
            top_level_dirs=top_level_dirs,
            depth=max_depth,
            most_files_dir=most_files_dir
        ),
        code_metrics=CodeMetrics(
            language_breakdown=language_breakdown,
            estimated_loc=estimated_loc,
            file_count_by_language=file_count_by_language
        )
    )
    
    logger.info(f"Repository analysis completed")
    return analysis

@Nodes.define(output="file_samples")
async def sample_key_files(repo_path: str, repo_analysis: RepoAnalysis) -> List[Dict[str, str]]:
    """Sample key files from the repository for LLM analysis.
    
    Args:
        repo_path: Path to the cloned repository
        repo_analysis: Repository analysis from analyze_repository
        
    Returns:
        List of dictionaries with file path and content
    """
    samples = []
    read_tool = ReadFileTool()
    
    # Try to find and sample key files
    key_file_patterns = [
        "README.md",
        "setup.py",
        "package.json",
        "requirements.txt",
        "Dockerfile",
        ".gitignore",
        "Makefile",
        "main.py",
        "index.js",
        "app.py",
        "server.js"
    ]
    
    # Sample files from each top-level directory
    for dir_name in repo_analysis.directory_structure.top_level_dirs:
        dir_path = os.path.join(repo_path, dir_name)
        if not os.path.isdir(dir_path):
            continue
            
        # Look for key files in this directory
        for root, _, files in os.walk(dir_path):
            if len(samples) >= 10:  # Limit number of samples
                break
                
            for file in files:
                if file in key_file_patterns or file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_path)
                    
                    try:
                        content = read_tool.execute(file_path=file_path)
                        samples.append({
                            "path": rel_path,
                            "content": content[:2000]  # Limit content size
                        })
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {str(e)}")
                        
                if len(samples) >= 10:  # Limit number of samples
                    break
    
    # If we don't have enough samples, add some of the largest files
    if len(samples) < 5:
        for file_info in repo_analysis.file_stats.largest_files:
            if len(samples) >= 10:
                break
                
            file_path = os.path.join(repo_path, file_info["path"])
            if os.path.isfile(file_path) and is_code_file(file_path):
                try:
                    content = read_tool.execute(file_path=file_path)
                    samples.append({
                        "path": file_info["path"],
                        "content": content[:2000]  # Limit content size
                    })
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
    
    logger.info(f"Sampled {len(samples)} key files from repository")
    return samples


@Nodes.structured_llm_node(
    system_prompt="""You are a world-class software architect, code quality expert, and technical leader with decades of experience across multiple programming languages and frameworks. Your analysis is known for being exceptionally detailed, insightful, and actionable.

Your task is to perform a comprehensive, expert-level analysis of a repository's structure, code quality, and architecture. Your analysis should be thorough, technical, and provide specific, actionable insights that would genuinely help improve the codebase.

For each section of your analysis:

1. ARCHITECTURE ASSESSMENT:
   - Evaluate the overall architectural patterns (MVC, MVVM, microservices, etc.)
   - Analyze component organization and separation of concerns
   - Assess modularity, coupling, and cohesion
   - Identify architectural strengths and weaknesses
   - Evaluate scalability considerations
   - Analyze dependency management and external integrations

2. CODE QUALITY INSIGHTS:
   - Assess code readability, consistency, and maintainability
   - Identify potential code smells and anti-patterns
   - Evaluate error handling and edge case management
   - Analyze naming conventions and code organization
   - Assess test coverage and testing strategies
   - Evaluate performance considerations and optimizations
   - Identify security concerns or best practices violations

3. MAIN TECHNOLOGIES:
   - Provide a detailed breakdown of frameworks, libraries, and tools
   - Assess version compatibility and dependency management
   - Evaluate technology choices against industry standards
   - Identify outdated dependencies or technologies that should be upgraded
   - Analyze build tools, CI/CD infrastructure, and deployment strategies

4. DOCUMENTATION ASSESSMENT:
   - Evaluate comprehensiveness, clarity, and accuracy
   - Assess code comments, inline documentation, and external docs
   - Identify gaps in documentation coverage
   - Analyze API documentation and examples
   - Evaluate onboarding documentation for new developers

5. KEY FEATURES:
   - Provide a detailed breakdown of the repository's main functionality
   - Analyze implementation approaches for key features
   - Assess the completeness of feature implementations
   - Identify unique or innovative aspects of the codebase
   - Evaluate feature organization and discoverability

6. IMPROVEMENT SUGGESTIONS:
   - Provide specific, actionable recommendations with examples
   - Prioritize suggestions based on impact and effort
   - Include code snippets or patterns that could be adopted
   - Suggest refactoring strategies with clear benefits
   - Recommend tools, libraries, or approaches that could enhance the codebase
   - Address technical debt with practical mitigation strategies

7. FINAL ASSESSMENT:
   - Provide a balanced, holistic evaluation
   - Highlight major strengths and areas for improvement
   - Assess overall code health and maintainability
   - Evaluate alignment with industry best practices
   - Provide a forward-looking perspective on the codebase's evolution

Your analysis should be deeply technical, specific, and actionable - not generic or superficial. Include specific examples from the codebase whenever possible. Focus on providing insights that would genuinely help improve the repository.""",
    output="llm_insights",
    response_model=LLMRepoInsights,
    prompt_template="""# Comprehensive Repository Analysis Request

## Repository Information
Repository Name: {{repo_name}}

## File Statistics
Total Files: {{repo_analysis.file_stats.total_files}}
Code Files: {{repo_analysis.file_stats.code_files_count}}
Documentation Files: {{repo_analysis.file_stats.documentation_files_count}}
Config Files: {{repo_analysis.file_stats.config_files_count}}

File Types:
{% for ext, count in repo_analysis.file_stats.file_types.items() %}
- {{ext}}: {{count}}
{% endfor %}

## Directory Structure
Top-level directories:
{% for dir in repo_analysis.directory_structure.top_level_dirs %}
- {{dir}}
{% endfor %}

Directory depth: {{repo_analysis.directory_structure.depth}}
Directory with most files: {{repo_analysis.directory_structure.most_files_dir}}

## Code Metrics
Language breakdown:
{% for lang, percentage in repo_analysis.code_metrics.language_breakdown.items() %}
- {{lang}}: {{percentage}}%
{% endfor %}

Estimated lines of code: {{repo_analysis.code_metrics.estimated_loc}}

## File Samples
{% if file_samples %}
Here are key files from the repository:

{% for file in file_samples %}
### {{file.path}}
```
{{file.content}}
```
{% endfor %}
{% endif %}

Based on this information, please provide an exceptionally detailed, technical, and actionable analysis of the repository. Your analysis should:

1. Go beyond surface-level observations
2. Provide specific, concrete examples from the code
3. Offer actionable, prioritized recommendations
4. Demonstrate deep technical expertise
5. Address both immediate improvements and long-term architectural considerations

For each section (Architecture, Code Quality, Technologies, Documentation, Features, Improvements, and Final Assessment), provide at least 3-5 detailed paragraphs with specific observations and recommendations. Include code examples, patterns, or anti-patterns where relevant.

Your analysis should be comprehensive enough to serve as a detailed code review and architectural assessment that would genuinely help improve the codebase.""",
    temperature=0.7,
    max_tokens=4000,  # Increased token limit for more detailed response
)
async def llm_analysis(
    model: str, 
    repo_name: str, 
    repo_analysis: RepoAnalysis,
    file_samples: List[Dict[str, str]]
) -> LLMRepoInsights:
    """Generate detailed LLM insights about the repository."""
    logger.debug(f"llm_analysis called with model: {model}")
    pass  # Implementation provided by the decorator

@Nodes.define(output="final_report")
async def generate_final_report(
    repo_name: str,
    repo_analysis: RepoAnalysis,
    llm_insights: LLMRepoInsights
) -> str:
    """Generate a final comprehensive report.
    
    Args:
        repo_name: Name of the repository
        repo_analysis: Repository analysis from analyze_repository
        llm_insights: LLM insights from llm_analysis
        
    Returns:
        Markdown string with the final report
    """
    # Format the report in markdown
    report = f"""# Repository Analysis Report: {repo_name}

## Repository Overview
- **Repository Name**: {repo_name}
- **Total Files**: {repo_analysis.file_stats.total_files}
- **Code Files**: {repo_analysis.file_stats.code_files_count}
- **Documentation Files**: {repo_analysis.file_stats.documentation_files_count}
- **Configuration Files**: {repo_analysis.file_stats.config_files_count}

## Directory Structure
- **Top-level Directories**: {', '.join(repo_analysis.directory_structure.top_level_dirs)}
- **Directory Depth**: {repo_analysis.directory_structure.depth}
- **Directory with Most Files**: {repo_analysis.directory_structure.most_files_dir}

## Code Metrics
### Language Breakdown
"""

    # Add language breakdown
    for lang, percentage in repo_analysis.code_metrics.language_breakdown.items():
        report += f"- **{lang}**: {percentage}%\n"
    
    report += f"\n- **Estimated Lines of Code**: {repo_analysis.code_metrics.estimated_loc}\n"
    
    # Add file types
    report += "\n### File Types\n"
    for ext, count in repo_analysis.file_stats.file_types.items():
        if count > 0:
            report += f"- **.{ext}**: {count} files\n"
    
    # Add LLM insights
    report += f"""
## Expert Analysis

### Architecture Assessment
{llm_insights.architecture_assessment}

### Code Quality Insights
{llm_insights.code_quality_insights}

### Main Technologies
{llm_insights.main_technologies}

### Documentation Assessment
{llm_insights.documentation_assessment}

### Key Features
{llm_insights.key_features}

### Improvement Suggestions
{llm_insights.improvement_suggestions}

## Final Assessment
{llm_insights.final_assessment}

---
*Report generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    return report

@Nodes.define(output=None)
async def save_report(final_report: str, repo_name: str) -> Dict[str, str]:
    """Save the final report to a file.
    
    Args:
        final_report: Final report from generate_final_report
        repo_name: Name of the repository
        
    Returns:
        Dictionary with the report file path
    """
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(os.path.expanduser("~"), "repo_analysis_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save the report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(reports_dir, f"{repo_name}_analysis_{timestamp}.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)
    
    logger.info(f"Report saved to {report_path}")
    return {"report_path": report_path}

@Nodes.define(output=None)
async def display_report(final_report: str, report_path: str) -> None:
    """Display the final report.
    
    Args:
        final_report: Final report from generate_final_report
        report_path: Path to the saved report
    """
    print("\n" + "="*50)
    print("📊 Repository Analysis Report 📊")
    print("="*50)
    print(f"Report saved to: {report_path}")
    print("\nReport Preview:")
    print("-"*50)
    
    # Print first 20 lines of the report
    lines = final_report.split("\n")
    preview_lines = lines[:20]
    print("\n".join(preview_lines))
    
    if len(lines) > 20:
        print("\n... (report continues)")
    
    print("-"*50)

# Define the Workflow
workflow = (
    Workflow("clone_repository")
    .add_observer(repo_analysis_progress_observer)
    .then("list_repository_contents")
    .then("analyze_repository")
    .then("sample_key_files")
    .then("llm_analysis")
    .then("generate_final_report")
    .then("save_report")
    .then("display_report")
)

def analyze_repository_flow(
    repo_url: str,
    auth_token: Optional[str] = None,
    model: str = "gemini/gemini-1.5-pro",
    branch: Optional[str] = None,
    task_id: str = "default",
    _handle_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
):
    """Analyze a Git repository and generate a comprehensive report.
    
    Args:
        repo_url: URL of the Git repository to analyze
        auth_token: Authentication token for private repositories
        model: LLM model to use for analysis
        task_id: Task ID for event handling
        _handle_event: Event handler function
        
    Returns:
        Dictionary with the workflow results
    """
    # Ensure template files exist
    ensure_templates_exist()
    
    # Ensure auth_token is a string
    auth_token_str = "" if auth_token is None else auth_token
    
    initial_context = {
        "repo_url": repo_url,
        "auth_token": auth_token_str,
        "model": model,
        "branch": branch,
    }

    logger.info(f"Starting repository analysis for {repo_url}")
    engine = workflow.build() 
    # Add the event observer if _handle_event is provided
    if _handle_event:
        # Create a lambda to bind task_id to the observer
        bound_observer = lambda event: asyncio.create_task(
            event_observer(event, task_id=task_id, _handle_event=_handle_event)
        )
        engine.add_observer(bound_observer)
    try:
        result = anyio.run(engine.run, initial_context)
        logger.info("Repository analysis completed successfully 🎉")
        return result
    except Exception as e:
        logger.error(f"Error in repository analysis workflow: {str(e)}")
        return {"error": str(e)}

def test_analysis(repo_url: str, auth_token: Optional[str] = None, model: str = "gemini/gemini-2.0-flash", branch: Optional[str] = None):
    """Direct test function for repository analysis without interactive prompts.
    
    Args:
        repo_url: URL of the Git repository to analyze
        auth_token: Optional authentication token for private repositories
        model: LLM model to use for analysis
        
    Returns:
        Result of the repository analysis
    """
    try:
        print(f"Analyzing repository: {repo_url}")
        print(f"Using model: {model}")
        print(f"Authentication: {'Provided' if auth_token else 'None'}")
        print("-" * 50)
        
        # Ensure auth_token is a string if provided, empty string if None
        auth_token_str = "" if auth_token is None else auth_token
        
        result = analyze_repository_flow(
            repo_url=repo_url,
            auth_token=auth_token_str,
            model=model,
            branch=branch
        )
        
        if result and "report_path" in result:
            print(f"\nFull report saved to: {result['report_path']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in repository analysis: {str(e)}")
        print(f"Error: {str(e)}")
        return None

def main():
    """Simple interactive test function for the repository analysis flow."""
    try:
        # Get repository URL with validation
        while True:
            repo_url = input("Enter the URL of the Git repository to analyze: ").strip()
            if not repo_url:
                print("Repository URL cannot be empty. Please try again.")
                continue
            
            if not (repo_url.startswith("https://github.com/") or 
                    repo_url.startswith("https://gitlab.com/")):
                confirm = input("URL doesn't seem to be from GitHub or GitLab. Continue anyway? (y/n): ").lower()
                if confirm != 'y':
                    continue
            break
        
        # Get optional auth token
        auth_token = input("Enter authentication token for private repositories (optional): ").strip() or None
        
        # Get model with default
        model = input("Enter LLM model to use (default: gemini/gemini-1.5-pro): ").strip()
        if not model:
            model = "gemini/gemini-1.5-pro"
        
        # Run the analysis
        logger.info(f"Starting repository analysis for {repo_url}")
        branch = input("Enter branch name (default: main): ").strip() or "main"
        result = analyze_repository_flow(
            repo_url=repo_url,
            auth_token=auth_token,
            model=model,
            branch=branch
        )
        
        # Show report path from result
        if result and "report_path" in result:
            print(f"\nFull report saved to: {result['report_path']}")
        
        return result
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return None
    except Exception as e:
        logger.error(f"Error in repository analysis: {str(e)}")
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Example 1: Analyze a public repository
    print("Example 1: Analyzing a public repository")
    test_analysis(
        repo_url="https://github.com/greatSumini/react-facebook-login",
        auth_token="",  # No token needed for public repos
        model="gemini/gemini-2.0-flash",
        branch="master"
    )
    
    # Example 2: Analyze a private repository (uncomment and add token)
    # print("\nExample 2: Analyzing a private repository")
    # test_analysis(
    #     repo_url="https://github.com/your-username/private-repo",
    #     auth_token="your-github-token",  # Add your token here
    #     model="gemini/gemini-2.0-flash"
    # )
    
    # Example 3: Use the interactive mode
    # print("\nExample 3: Interactive mode")
    # main()
