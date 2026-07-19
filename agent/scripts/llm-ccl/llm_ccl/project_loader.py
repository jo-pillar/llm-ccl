from __future__ import annotations

import importlib
import pkgutil

from .models import ExperimentSpec


def discover_projects(package_name: str = "llm_ccl.projects") -> dict[str, ExperimentSpec]:
    package = importlib.import_module(package_name)
    projects: dict[str, ExperimentSpec] = {}
    for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_info.name)
        project = getattr(module, "PROJECT", None)
        if not isinstance(project, ExperimentSpec):
            raise ValueError(f"{module_info.name} must export PROJECT: ExperimentSpec")
        if project.name in projects:
            raise ValueError(f"duplicate project name: {project.name}")
        projects[project.name] = project
    return dict(sorted(projects.items()))

