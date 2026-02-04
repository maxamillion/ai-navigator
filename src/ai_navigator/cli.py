"""CLI entry point for AI Navigator agents."""

import argparse
import sys

import uvicorn


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ai-navigator",
        description="AI Navigator - Kubernetes-native Supervisor/Sub-Agent system",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run agent command
    run_parser = subparsers.add_parser("run", help="Run an agent")
    run_parser.add_argument(
        "agent",
        choices=["supervisor", "model-catalog", "resource-provisioning", "deployment-monitor"],
        help="Agent to run",
    )
    run_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    run_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    run_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    # Operator command
    operator_parser = subparsers.add_parser("operator", help="Run the Kubernetes operator")
    operator_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.command == "run":
        return run_agent(args)
    elif args.command == "operator":
        return run_operator(args)
    else:
        parser.print_help()
        return 1


def run_agent(args: argparse.Namespace) -> int:
    """Run an agent server."""
    agent_modules = {
        "supervisor": "ai_navigator.agents.supervisor.agent:app",
        "model-catalog": "ai_navigator.agents.model_catalog.agent:app",
        "resource-provisioning": "ai_navigator.agents.resource_provisioning.agent:app",
        "deployment-monitor": "ai_navigator.agents.deployment_monitor.agent:app",
    }

    module = agent_modules.get(args.agent)
    if not module:
        print(f"Unknown agent: {args.agent}", file=sys.stderr)
        return 1

    print(f"Starting {args.agent} agent on {args.host}:{args.port}")

    uvicorn.run(
        module,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

    return 0


def run_operator(args: argparse.Namespace) -> int:
    """Run the Kubernetes operator."""
    import kopf

    kopf_args = ["run", "operator/main.py"]
    if args.verbose:
        kopf_args.append("--verbose")

    # Run kopf with the operator module
    try:
        kopf.run()
    except Exception as e:
        print(f"Operator error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
