from __future__ import annotations

import importlib
import json
import sys
from typing import Any

try:
    import click

    _CLICK_AVAILABLE = True
except ImportError:
    _CLICK_AVAILABLE = False


if _CLICK_AVAILABLE:

    @click.group()
    def cli() -> None:
        """predikit: ML model utilities for LLM agents."""

    @cli.command()
    @click.argument("model_path", type=click.Path(exists=True))
    @click.option(
        "--name", default="model", show_default=True, help="Tool name used in schema generation."
    )
    @click.option(
        "--description", default="ML model prediction", show_default=True, help="Tool description."
    )
    def inspect(model_path: str, name: str, description: str) -> None:
        """Inspect a saved model file and print its metadata and OpenAI schema."""
        try:
            import joblib
        except ImportError as err:
            raise click.ClickException(
                "joblib is required. Install it with: pip install predikit[cli]"
            ) from err

        from pydantic import create_model

        from .introspect import introspect
        from .tool import ModelTool

        model = joblib.load(model_path)
        meta = introspect(model)

        click.echo(f"Model:    {type(model).__name__}")
        click.echo(f"Task:     {meta['task']}")
        if meta["n_features"] is not None:
            click.echo(f"Features: {meta['n_features']}")

        if meta["feature_names"]:
            click.echo("Feature names:")
            for fname in meta["feature_names"]:
                click.echo(f"  {fname}")
        else:
            click.echo("Feature names: (none - fit the model with a named DataFrame to enable)")

        if meta["classes"] is not None:
            click.echo(f"Classes:  {meta['classes']}")

        if meta["feature_names"]:
            fields: dict[str, Any] = dict.fromkeys(meta["feature_names"], (float, ...))
            input_schema = create_model("Input", **fields)
            tool = ModelTool(
                model=model,
                name=name,
                description=description,
                input_schema=input_schema,
                output_name="prediction",
                output_description="model output",
            )
            click.echo("\nOpenAI schema:")
            click.echo(json.dumps(tool.to_openai(), indent=2))
        else:
            click.echo("\nOpenAI schema: unavailable (fit model with a named DataFrame to enable)")

    @cli.command()
    @click.argument("registry_target")
    @click.option("--name", default="predikit", show_default=True, help="MCP server name.")
    @click.option(
        "--transport",
        type=click.Choice(["stdio", "streamable-http"]),
        default="stdio",
        show_default=True,
        help="MCP transport to run.",
    )
    @click.option(
        "--host", default=None, help="Bind host for HTTP transports. [default: 127.0.0.1]"
    )
    @click.option(
        "--port", type=int, default=None, help="Bind port for HTTP transports. [default: 8000]"
    )
    def serve(
        registry_target: str,
        name: str,
        transport: str,
        host: str | None,
        port: int | None,
    ) -> None:
        """Serve a ToolRegistry from MODULE:ATTRIBUTE over MCP.

        ATTRIBUTE may be a ToolRegistry instance or a zero-argument factory.
        """
        if ":" not in registry_target:
            raise click.ClickException("REGISTRY_TARGET must use the form MODULE:ATTRIBUTE")
        module_name, attribute_name = registry_target.split(":", 1)

        # Keep this block narrow: it resolves the target only. Errors raised while
        # building or running the server must surface with their own traceback
        # rather than be reported as a bad registry target.
        try:
            module = importlib.import_module(module_name)
        except ImportError as err:
            raise click.ClickException(f"Could not import module '{module_name}': {err}") from err
        try:
            registry = getattr(module, attribute_name)
        except AttributeError as err:
            raise click.ClickException(
                f"Module '{module_name}' has no attribute '{attribute_name}'"
            ) from err

        if callable(registry):
            registry = registry()

        from .mcp import create_mcp_server

        server = create_mcp_server(registry, name=name, host=host, port=port)
        server.run(transport=transport)

else:

    def cli() -> None:  # type: ignore[misc]
        """Fallback when click is not installed."""
        print(
            "Error: 'click' is required. Install it with: pip install predikit[cli]",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    cli()
