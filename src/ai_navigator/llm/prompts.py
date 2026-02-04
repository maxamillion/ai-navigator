"""Prompt templates for LLM-based operations."""

from typing import Any


class PromptTemplates:
    """Collection of prompt templates for AI Navigator."""

    SYSTEM_SUPERVISOR = """You are the AI Navigator Supervisor Agent for OpenShift AI.
Your role is to help users deploy and manage AI models on OpenShift AI.

You have access to the following sub-agents:
1. Model Catalog Agent - Query models, get benchmarks, TrustyAI scores
2. Resource Provisioning Agent - Generate deployments, estimate costs, validate SLOs
3. Deployment Monitor Agent - Check status, query metrics, monitor SLOs

When users ask questions, decompose their request into sub-tasks and delegate to the appropriate agents.
Always provide helpful, accurate information about model deployment on OpenShift AI.

Available skills:
- recommend_model: Get model recommendations for a workload
- deploy_model: Deploy a model to the cluster
- check_status: Check deployment status and health
- list_models: List available models in the registry"""

    TASK_DECOMPOSITION = """Analyze the user request and decompose it into sub-tasks.

User Request: {user_request}

Available Sub-Agents and their skills:
1. model_catalog:
   - query_models: Search for models
   - get_model_details: Get model information
   - get_benchmarks: Get performance benchmarks
   - get_trustyai_scores: Get safety evaluations
   - recommend_for_workload: Recommend models for a use case

2. resource_provisioning:
   - generate_deployment_config: Create KServe config
   - estimate_cost: Estimate deployment costs
   - validate_slo_compliance: Validate against SLOs
   - apply_deployment: Deploy to cluster
   - generate_guardrails: Create TrustyAI guardrails

3. deployment_monitor:
   - get_deployment_status: Check current status
   - query_metrics: Query Prometheus metrics
   - check_slo_violations: Check for SLO issues
   - get_pod_logs: Get container logs
   - get_health_summary: Get overall health

Respond with a JSON object containing:
{{
    "intent": "Brief description of user intent",
    "subtasks": [
        {{
            "id": "1",
            "agent": "agent_name",
            "skill": "skill_id",
            "params": {{}},
            "depends_on": []
        }}
    ],
    "parallel_groups": [[task_ids that can run in parallel]]
}}

Only output the JSON, no other text."""

    RESULT_AGGREGATION = """Synthesize the results from multiple sub-agents into a coherent response.

User Request: {user_request}

Sub-Agent Results:
{results}

Create a natural, helpful response that:
1. Directly addresses the user's request
2. Summarizes key findings from each sub-agent
3. Provides actionable recommendations
4. Includes relevant details without overwhelming

Format the response using markdown for readability.
Be concise but comprehensive."""

    INTENT_CLASSIFICATION = """Classify the user's intent and identify the primary action needed.

User Message: {message}

Respond with a JSON object:
{{
    "intent": "one of: recommend, deploy, monitor, query, configure, troubleshoot",
    "entities": {{
        "model_name": "extracted model name or null",
        "workload_type": "extracted workload type or null",
        "namespace": "extracted namespace or null"
    }},
    "confidence": 0.0 to 1.0
}}

Only output the JSON."""

    SKILL_SELECTION = """Select the most appropriate skill for the user request.

User Request: {user_request}
Current Agent: {agent_name}

Available Skills:
{skills}

Respond with a JSON object:
{{
    "skill_id": "selected skill id",
    "params": {{extracted parameters}},
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}

Only output the JSON."""

    @classmethod
    def format_decomposition(cls, user_request: str) -> str:
        """Format task decomposition prompt."""
        return cls.TASK_DECOMPOSITION.format(user_request=user_request)

    @classmethod
    def format_aggregation(cls, user_request: str, results: list[dict[str, Any]]) -> str:
        """Format result aggregation prompt."""
        results_text = ""
        for i, result in enumerate(results, 1):
            results_text += f"\n### Result {i} (from {result.get('agent', 'unknown')}):\n"
            results_text += result.get("message", str(result.get("data", {})))
            results_text += "\n"

        return cls.RESULT_AGGREGATION.format(
            user_request=user_request,
            results=results_text,
        )

    @classmethod
    def format_intent_classification(cls, message: str) -> str:
        """Format intent classification prompt."""
        return cls.INTENT_CLASSIFICATION.format(message=message)

    @classmethod
    def format_skill_selection(
        cls,
        user_request: str,
        agent_name: str,
        skills: list[dict[str, Any]],
    ) -> str:
        """Format skill selection prompt."""
        skills_text = ""
        for skill in skills:
            skills_text += f"\n- **{skill['id']}** ({skill['name']}): {skill['description']}"

        return cls.SKILL_SELECTION.format(
            user_request=user_request,
            agent_name=agent_name,
            skills=skills_text,
        )
