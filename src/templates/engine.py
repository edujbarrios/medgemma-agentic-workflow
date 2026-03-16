"""Jinja2 template engine for prompt rendering."""

from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, BaseLoader, TemplateNotFound
from loguru import logger
import os


class TemplateEngine:
    """Jinja2-based template engine for prompt rendering."""
    
    def __init__(
        self,
        template_dir: Optional[str] = None,
        auto_reload: bool = True,
        trim_blocks: bool = True,
        lstrip_blocks: bool = True,
    ):
        """Initialize template engine.
        
        Args:
            template_dir: Directory containing templates
            auto_reload: Auto reload templates on change
            trim_blocks: Trim blocks in templates
            lstrip_blocks: Left strip blocks in templates
        """
        self.template_dir = template_dir or self._find_template_dir()
        
        if self.template_dir and Path(self.template_dir).exists():
            loader = FileSystemLoader(self.template_dir)
        else:
            loader = BaseLoader()
        
        self.env = Environment(
            loader=loader,
            auto_reload=auto_reload,
            trim_blocks=trim_blocks,
            lstrip_blocks=lstrip_blocks,
        )
        
        # Add custom filters
        self._add_custom_filters()
        
        logger.info(f"Initialized TemplateEngine with dir: {self.template_dir}")
    
    @staticmethod
    def _find_template_dir() -> Optional[str]:
        """Find templates directory in project structure.
        
        Returns:
            Path to templates directory or None
        """
        # Check current working directory
        for possible_dir in [
            "templates",
            "./templates",
            "../templates",
            "src/templates",
            "./src/templates",
        ]:
            path = Path(possible_dir)
            if path.exists() and path.is_dir():
                return str(path.absolute())
        
        return None
    
    def _add_custom_filters(self) -> None:
        """Add custom Jinja2 filters."""
        
        def format_list(items, separator=", "):
            """Format list as string."""
            if isinstance(items, (list, tuple)):
                return separator.join(str(i) for i in items)
            return str(items)
        
        def format_dict(d, indent=0):
            """Format dictionary."""
            if not isinstance(d, dict):
                return str(d)
            lines = []
            for k, v in d.items():
                if isinstance(v, dict):
                    lines.append(f"{' ' * indent}{k}:\\n{format_dict(v, indent + 2)}")
                elif isinstance(v, (list, tuple)):
                    lines.append(f"{' ' * indent}{k}: {format_list(v)}")
                else:
                    lines.append(f"{' ' * indent}{k}: {v}")
            return "\\n".join(lines)
        
        def truncate(s, length=100, end="..."):
            """Truncate string."""
            if len(str(s)) <= length:
                return str(s)
            return str(s)[:length - len(end)] + end
        
        def upper_first(s):
            """Capitalize first letter."""
            s = str(s)
            return s[0].upper() + s[1:] if s else s
        
        # Register filters
        self.env.filters['format_list'] = format_list
        self.env.filters['format_dict'] = format_dict
        self.env.filters['truncate'] = truncate
        self.env.filters['upper_first'] = upper_first
    
    def render(
        self,
        template_name: str,
        variables: Dict[str, Any],
        from_string: bool = False
    ) -> str:
        """Render template with variables.
        
        Args:
            template_name: Template name or string
            variables: Template variables
            from_string: If True, treat template_name as template string
            
        Returns:
            Rendered template
        """
        try:
            if from_string:
                template = self.env.from_string(template_name)
            else:
                # Check if file exists
                template_path = Path(template_name)
                if not template_path.is_absolute():
                    if self.template_dir:
                        template_path = Path(self.template_dir) / template_name
                
                if template_path.exists():
                    # Load from file
                    with open(template_path, 'r', encoding='utf-8') as f:
                        template_str = f.read()
                    template = self.env.from_string(template_str)
                else:
                    # Try to load from registered loader
                    template = self.env.get_template(template_name)
            
            return template.render(**variables)
        
        except TemplateNotFound:
            logger.error(f"Template not found: {template_name}")
            raise
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            raise
    
    def render_file(
        self,
        file_path: str,
        variables: Dict[str, Any]
    ) -> str:
        """Render template from file.
        
        Args:
            file_path: Path to template file
            variables: Template variables
            
        Returns:
            Rendered output
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")
        
        return self.render(file_path, variables)
    
    def render_string(
        self,
        template_string: str,
        variables: Dict[str, Any]
    ) -> str:
        """Render template from string.
        
        Args:
            template_string: Template string
            variables: Template variables
            
        Returns:
            Rendered output
        """
        return self.render(template_string, variables, from_string=True)


def render_template(
    template_string: str,
    variables: Dict[str, Any],
    template_dir: Optional[str] = None
) -> str:
    """Convenience function to render template string.
    
    Args:
        template_string: Template string
        variables: Template variables
        template_dir: Optional template directory
        
    Returns:
        Rendered string
    """
    engine = TemplateEngine(template_dir=template_dir)
    return engine.render_string(template_string, variables)
