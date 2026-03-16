# Contributing

Contributions are welcome! This document explains how to contribute to the MedGemma Agents project.

## Code of Conduct

- Be respectful and inclusive
- Focus on the code, not on personal criticism
- Help maintain a welcoming environment

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/medgemma-agents.git
cd medgemma-agents
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/bug-description
```

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8
- **Formatting**: Use Black (100 char line length)
- **Linting**: Use Ruff
- **Type Hints**: Use type annotations where possible
- **Docstrings**: Use Google-style docstrings

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/ --fix

# Type check
mypy src/

# Run tests
pytest tests/ --cov=src
```

### Testing

- Write tests for new features
- Maintain test coverage above 80%
- Test edge cases and error conditions
- Use descriptive test names

Example:
```python
def test_clinical_agent_processes_valid_input():
    """Test that clinical agent successfully processes valid input."""
    # Test implementation
    pass
```

### Documentation

- Update README.md if adding user-facing features
- Update ARCHITECTURE.md for structural changes
- Add docstrings to all public methods
- Include examples for new features

## Submitting Changes

### 1. Commit Guidelines

```bash
# Make focused commits
git add <specific files>
git commit -m "Brief description of change

Optional longer explanation if needed."
```

**Commit Message Format**:
- Use imperative mood ("Add feature" not "Added feature")
- Start with a verb: Add, Fix, Update, Remove, Refactor, etc.
- First line max 50 characters
- Reference issues: "Fixes #123"

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

On GitHub:
1. Click "New Pull Request"
2. Provide clear description of changes
3. Reference relevant issues
4. Ensure tests pass

**PR Title Format**:
```
[Type] Brief description

where [Type] is one of:
- [Feature] for new functionality
- [Fix] for bug fixes
- [Docs] for documentation
- [Refactor] for code refactoring
- [Test] for test additions
```

### 4. Code Review

- Respond to reviewer comments
- Make requested changes
- Re-request review after updates
- Be patient and professional

## Types of Contributions

### Bug Reports

```bash
# Create issue with:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version, etc.)
```

### Feature Requests

```bash
# Create issue with:
- Use case description
- Proposed solution
- Alternatives considered
- Example usage
```

### Documentation

- Fix typos and clarify unclear sections
- Add missing documentation
- Improve examples
- Translate documentation

### Code

- New agents (clinical, reasoning, etc.)
- Workflow improvements
- Template engine enhancements
- Configuration options
- Testing improvements

## Project Structure Conventions

```
src/
  module/
    __init__.py          # Public API
    implementation.py    # Implementation
    tests/              # Optional unit tests

tests/
  test_module.py        # Test modules matching structure

docs/
  FEATURE_NAME.md       # Feature documentation

examples/
  feature_example.py    # Usage examples
```

## Adding New Agents

```python
# src/agents/my_agent.py
from medgemma_agents.agents import BaseAgent

class MyAgent(BaseAgent):
    """Description of agent."""
    
    def process(self, input_data):
        """Process input data."""
        # Implementation
        pass

# Add to src/agents/__init__.py
from medgemma_agents.agents.my_agent import MyAgent
```

## Adding New Agents

```python
# src/agents/my_agent.py
from medgemma_agents.agents import BaseAgent

class MyAgent(BaseAgent):
    """Description of agent."""
    
    def process(self, input_data):
        """Process input data."""
        # Implementation
        pass

# Add to src/agents/__init__.py
from medgemma_agents.agents.my_agent import MyAgent
```

## Performance

- Profile code before optimizing
- Consider memory usage
- Minimize API calls
- Cache when appropriate

## Security

- Never commit API keys
- Use environment variables
- Validate all inputs
- Sanitize sensitive data in logs

## Getting Help

- Check existing issues/discussions
- Review documentation
- Ask in PR comments
- Create a discussion for ideas

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Credited in release notes
- Recognized in project README

---

Thank you for contributing to MedGemma Agents! 🙏
