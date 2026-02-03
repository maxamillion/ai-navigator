# AI Navigator Documentation

Welcome to the AI Navigator documentation. This index helps you find the right document for your needs.

## Quick Links

| I want to... | Read this |
|--------------|-----------|
| Get started quickly | [Onboarding Guide](ONBOARDING.md) |
| Understand the system design | [Architecture](ARCHITECTURE.md) |
| Set up my development environment | [Developer Guide](DEVELOPER_GUIDE.md) |
| Use the API | [API Reference](API_REFERENCE.md) |
| Look up a specific module | [Module Reference](MODULE_REFERENCE.md) |

---

## Documentation Overview

### [README.md](../README.md)
**Audience:** All developers
**Purpose:** Project overview, quick start, basic usage

Contents:
- Features
- Installation
- Quick start
- Project structure
- Configuration

### [ONBOARDING.md](ONBOARDING.md)
**Audience:** New team members
**Purpose:** Get productive in your first week

Contents:
- Day 1 setup checklist
- Day 2 codebase exploration
- Week 1 starter tasks
- Quick reference
- Cheatsheet

### [ARCHITECTURE.md](ARCHITECTURE.md)
**Audience:** Developers needing to understand the system
**Purpose:** Detailed system architecture and design decisions

Contents:
- High-level architecture
- Core components
- 8-stage workflow engine
- Data flow diagrams
- State management
- External integrations
- Design patterns
- Extension points

### [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
**Audience:** Active contributors
**Purpose:** Day-to-day development workflow

Contents:
- Development setup
- Testing guide
- API development
- Adding features
- Code style
- Debugging
- Troubleshooting

### [API_REFERENCE.md](API_REFERENCE.md)
**Audience:** API users and frontend developers
**Purpose:** Complete API documentation

Contents:
- All endpoints
- Request/response schemas
- Error handling
- Data types
- Examples

### [MODULE_REFERENCE.md](MODULE_REFERENCE.md)
**Audience:** Developers working on specific modules
**Purpose:** Detailed reference for all classes and functions

Contents:
- Models (workflow, capacity, deployment)
- Workflow engine and stages
- Capacity planning
- MCP integration
- Model Registry
- Deployment generation
- State management
- Quickstarts
- Configuration

### [CLAUDE.md](../CLAUDE.md)
**Audience:** All contributors
**Purpose:** Development principles and project guidelines

Contents:
- Build and test commands
- Development principles
- Pythonic practices
- Architecture overview

---

## Document Maintenance

When updating code, remember to update relevant documentation:

| Change Type | Update These Docs |
|-------------|-------------------|
| New API endpoint | API_REFERENCE.md |
| New workflow stage | ARCHITECTURE.md, MODULE_REFERENCE.md |
| New model field | MODULE_REFERENCE.md |
| Configuration change | README.md, ARCHITECTURE.md |
| New dependency | README.md, DEVELOPER_GUIDE.md |

## Contributing to Documentation

1. Keep documentation close to the code it describes
2. Use clear, concise language
3. Include code examples where helpful
4. Keep tables and lists for easy scanning
5. Update the index when adding new documents
