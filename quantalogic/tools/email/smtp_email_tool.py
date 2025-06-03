"""Tool for sending emails using SMTP protocol."""

import os
import smtplib
import base64
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger
from quantalogic.tools.tool import Tool, ToolArgument


class SmtpEmailTool(Tool):
    """Tool for sending emails using SMTP protocol with support for attachments."""

    name: str = "smtp_email_tool"
    description: str = (
        "Sends an email using SMTP protocol. Currently supports Gmail SMTP server. "
        "Can send to multiple recipients and attach files. "
        "Requires SMTP credentials to be provided. "
        "Supports variable interpolation with $variable$ syntax if need_variables is True."
    )
    need_validation: bool = True  # Require validation before sending emails
    need_variables: bool = True   # Support variable interpolation
    agent_id: Optional[str] = None
    
    arguments: list = [
        ToolArgument(
            name="sender_email",
            arg_type="string",
            description="The email address of the sender.",
            required=True,
            example="sender@gmail.com",
        ),
        ToolArgument(
            name="auth_token",
            arg_type="string",
            description="OAuth2 token for authentication.",
            required=True,
            example="ya29.a0AfB_byC...",
        ),
        ToolArgument(
            name="receiver_email",
            arg_type="string",
            description="The email address(es) of the recipient(s). For multiple recipients, separate with commas.",
            required=True,
            example="recipient@example.com",
        ),
        ToolArgument(
            name="subject",
            arg_type="string",
            description="The subject of the email.",
            required=True,
            example="Important Information",
        ),
        ToolArgument(
            name="content",
            arg_type="string",
            description="""
            The content of the email. Can be plain text or HTML.
            Use CDATA to escape special characters if needed.
            """,
            required=True,
            example="Hello, this is the content of the email.",
        ),
        ToolArgument(
            name="is_html",
            arg_type="string",
            description="If true, the content will be treated as HTML. Defaults to False.",
            required=False,
            default="False",
            example="True",
        ),
        ToolArgument(
            name="attachment_paths",
            arg_type="string",
            description="""
            Comma-separated list of file paths to attach to the email.
            Paths should be absolute or relative to the current working directory.
            """,
            required=False,
            example="/path/to/file1.pdf,/path/to/file2.jpg",
        ),
        ToolArgument(
            name="smtp_server",
            arg_type="string",
            description="The SMTP server to use. Defaults to Gmail's SMTP server.",
            required=False,
            default="smtp.gmail.com",
            example="smtp.gmail.com",
        ),
        ToolArgument(
            name="smtp_port",
            arg_type="string",
            description="The SMTP server port to use. Defaults to 587 (TLS).",
            required=False,
            default="587",
            example="587",
        ),
        ToolArgument(
            name="variables",
            arg_type="string",
            description="Variables to interpolate in the content. Will be automatically provided when need_variables=True.",
            required=False,
            example="{}",
        ),
    ]

    def _validate_email_format(self, email: str) -> bool:
        """Basic validation for email format.
        
        Args:
            email (str): Email address to validate
            
        Returns:
            bool: True if email format is valid, False otherwise
        """
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _validate_file_paths(self, file_paths: List[str]) -> List[str]:
        """Validate that file paths exist and are accessible.
        
        Args:
            file_paths (List[str]): List of file paths to validate
            
        Returns:
            List[str]: List of valid file paths
            
        Raises:
            ValueError: If any file path is invalid or inaccessible
        """
        valid_paths = []
        for path in file_paths:
            path = path.strip()
            if not path:
                continue
                
            file_path = Path(path)
            if not file_path.exists():
                raise ValueError(f"File not found: {path}")
            if not file_path.is_file():
                raise ValueError(f"Not a file: {path}")
            if not os.access(file_path, os.R_OK):
                raise ValueError(f"File not readable: {path}")
                
            valid_paths.append(str(file_path))
            
        return valid_paths
    
    def _interpolate_variables(self, content: str, variables: dict = None) -> str:
        """Interpolate variables in the content using $var$ syntax.
        
        Args:
            content (str): The content that may contain variable references
            variables (dict, optional): Dictionary of variables to interpolate
            
        Returns:
            str: Content with variables interpolated
        """
        if not isinstance(content, str) or not variables:
            return content
            
        try:
            import re
            
            # Interpolate each variable in the content
            for var_name, var_value in variables.items():
                if not var_name.startswith('$'):
                    # Create pattern for $var_name$
                    pattern = f"\\${re.escape(var_name)}\\$"
                    # Replace with the variable value
                    content = re.sub(pattern, str(var_value), content)
                    
            return content
        except Exception as e:
            logger.error(f"Error in _interpolate_variables: {str(e)}")
            return content

    def _perform_oauth2_authentication(self, server: smtplib.SMTP, user: str, auth_token: str) -> None:
        """Authenticate to the SMTP server using OAuth2.
        
        Args:
            server: SMTP server instance
            user: Email address of the user
            auth_token: OAuth2 token for authentication
            
        Raises:
            smtplib.SMTPAuthenticationError: If authentication fails
        """
        auth_string = f"user={user}\1auth=Bearer {auth_token}\1\1"
        auth_string_b64 = base64.b64encode(auth_string.encode()).decode()
        server.docmd("AUTH", f"XOAUTH2 {auth_string_b64}")
    
    def execute(
        self, 
        sender_email: str, 
        receiver_email: str, 
        subject: str, 
        content: str, 
        auth_token: str,
        is_html: str = "False", 
        attachment_paths: str = "", 
        smtp_server: str = "smtp.gmail.com", 
        smtp_port: str = "587",
        agent_id: str = None, 
        variables: Dict = None
    ) -> str:
        """Sends an email using SMTP protocol with OAuth2 authentication.

        Args:
            sender_email (str): The email address of the sender.
            receiver_email (str): The email address(es) of the recipient(s). For multiple recipients, separate with commas.
            subject (str): The subject of the email.
            content (str): The content of the email. Can be plain text or HTML.
            auth_token (str): OAuth2 token for authentication.
            is_html (str, optional): If "True", the content will be treated as HTML. Defaults to "False".
            attachment_paths (str, optional): Comma-separated list of file paths to attach to the email.
            smtp_server (str, optional): The SMTP server to use. Defaults to "smtp.gmail.com".
            smtp_port (str, optional): The SMTP server port to use. Defaults to "587".
            agent_id (str, optional): The agent ID. Defaults to None.
            variables (Dict, optional): Variables to interpolate in the content. Defaults to None.

        Returns:
            str: Status message indicating success or failure.

        Raises:
            ValueError: If email format is invalid or files are not accessible.
            Exception: For SMTP connection or authentication errors.
        """
        try:
            # Validate email format
            if not self._validate_email_format(sender_email):
                raise ValueError(f"Invalid sender email format: {sender_email}")
                
            # Process receiver emails (can be comma-separated)
            receivers = [email.strip() for email in receiver_email.split(',') if email.strip()]
            for email in receivers:
                if not self._validate_email_format(email):
                    raise ValueError(f"Invalid receiver email format: {email}")
            
            if not receivers:
                raise ValueError("No valid receiver email addresses provided")
                
            # Interpolate variables in subject and content if needed
            if variables:
                subject = self._interpolate_variables(subject, variables)
                content = self._interpolate_variables(content, variables)
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(receivers)
            msg['Subject'] = subject
            
            # Attach content
            content_type = 'html' if is_html.lower() == 'true' else 'plain'
            msg.attach(MIMEText(content, content_type))
            
            # Process attachments if any
            if attachment_paths:
                file_paths = [path.strip() for path in attachment_paths.split(',') if path.strip()]
                valid_paths = self._validate_file_paths(file_paths)
                
                for file_path in valid_paths:
                    with open(file_path, 'rb') as file:
                        part = MIMEApplication(file.read(), Name=os.path.basename(file_path))
                        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                        msg.attach(part)
            
            # Validate authentication token
            if not auth_token:
                raise ValueError("OAuth2 authentication requires an auth_token parameter")
                
            # Connect to SMTP server and send email
            with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
                server.starttls()  # Secure the connection
                
                # Authenticate using OAuth2
                self._perform_oauth2_authentication(server, sender_email, auth_token)
                    
                server.send_message(msg)
                
            logger.info(f"Email sent successfully to {', '.join(receivers)}")
            return f"Email sent successfully to {', '.join(receivers)}"
            
        except ValueError as ve:
            error_msg = f"Validation error: {str(ve)}"
            logger.error(error_msg)
            return error_msg
            
        except smtplib.SMTPAuthenticationError:
            error_msg = "SMTP authentication failed. Check your email and password. For Gmail, you may need to use an App Password."
            logger.error(error_msg)
            return error_msg
            
        except smtplib.SMTPException as se:
            error_msg = f"SMTP error: {str(se)}"
            logger.error(error_msg)
            return error_msg
            
        except Exception as e:
            error_msg = f"Error sending email: {str(e)}"
            logger.error(error_msg)
            return error_msg


if __name__ == "__main__":
    # Print the tool documentation
    tool = SmtpEmailTool()
    print(tool.to_markdown())
    print("\n" + "-"*50 + "\n")
    
    # Test configuration
    # Replace these values with your actual test credentials
    sender_email = "yatchi.leet@gmail.com"  # Replace with your Gmail address
    receiver_email = "yatchi.leet2@gmail.com"  # Replace with recipient email
    
    # Authentication with OAuth2 token
    auth_token = ""  # Replace with your OAuth2 token
    
    # Email content
    subject = "Test Email from SmtpEmailTool"
    content = """Hello,
    
    This is a test email sent from the SmtpEmailTool in the QuantaLogic agent.
    
    Best regards,
    QuantaLogic Team
    """
    
    # Optional settings
    is_html = "False"  # Set to "True" to send as HTML
    attachment_paths = ""  # Add comma-separated file paths if needed
    smtp_server = "smtp.gmail.com"
    smtp_port = "587"
    
    # Instructions for testing
    print("To test this tool:")
    print("1. Edit this file to set your email credentials")
    print("2. For Gmail, use an App Password (https://myaccount.google.com/apppasswords)")
    print("3. Uncomment the execute_test() call at the bottom of this file")
    print("\n" + "-"*50 + "\n")
    
    def execute_test():
        """Execute the email sending test with the configured values."""
        print(f"Sending test email from {sender_email} to {receiver_email}...")
        
        try:
            result = tool.execute(
                sender_email=sender_email,
                receiver_email=receiver_email,
                subject=subject,
                content=content,
                auth_token=auth_token,
                is_html=is_html,
                attachment_paths=attachment_paths,
                smtp_server=smtp_server,
                smtp_port=smtp_port
            )
            
            print(f"\nResult: {result}")
            
        except Exception as e:
            print(f"\nError during test: {str(e)}")
    
    # Run the test with the configured credentials
    execute_test()
