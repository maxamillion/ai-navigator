"""Tests for capacity planning."""

import pytest
from ai_navigator.planning.capacity import CapacityPlanner
from ai_navigator.planning.recommender import ModelRecommender
from ai_navigator.planning.whatif import WhatIfAnalyzer
from ai_navigator.models.capacity import BenchmarkData, WhatIfScenario
from ai_navigator.models.workflow import (
    TrafficProfile,
    TrafficPattern,
    SLORequirements,
    ModelRequirements,
)


class TestCapacityPlanner:
    """Test capacity planner."""

    @pytest.fixture
    def planner(self):
        """Create capacity planner."""
        return CapacityPlanner()

    @pytest.fixture
    def benchmark(self):
        """Create sample benchmark data."""
        return BenchmarkData(
            model_name="llama-2-7b",
            model_version="1.0",
            gpu_type="A100-40GB",
            gpu_count=1,
            p50_latency_ms=150,
            p95_latency_ms=250,
            p99_latency_ms=400,
            tokens_per_second=120,
            requests_per_second=15,
            gpu_memory_gb=14,
            gpu_utilization_percent=85,
        )

    @pytest.fixture
    def traffic(self):
        """Create sample traffic profile."""
        return TrafficProfile(
            pattern=TrafficPattern.STEADY,
            requests_per_second=30,
            average_input_tokens=512,
            average_output_tokens=256,
        )

    @pytest.fixture
    def slo(self):
        """Create sample SLO requirements."""
        return SLORequirements(
            p50_latency_ms=500,
            p95_latency_ms=1000,
            p99_latency_ms=2000,
            availability_percent=99.9,
        )

    def test_calculate_capacity_plan(self, planner, benchmark, traffic, slo):
        """Test calculating capacity plan."""
        plan = planner.calculate_capacity_plan(benchmark, traffic, slo)

        assert plan.model_name == "llama-2-7b"
        assert plan.min_replicas >= 2  # At least 2 for HA
        assert plan.max_replicas >= plan.min_replicas
        assert plan.gpu_type == "A100-40GB"

    def test_plan_meets_slo(self, planner, benchmark, traffic, slo):
        """Test that plan meets SLO requirements."""
        plan = planner.calculate_capacity_plan(benchmark, traffic, slo)

        assert plan.meets_slo
        assert len(plan.slo_violations) == 0

    def test_plan_slo_violations(self, planner, benchmark, traffic):
        """Test plan with SLO violations."""
        # Create tight SLO that benchmark can't meet
        tight_slo = SLORequirements(
            p50_latency_ms=50,  # Tighter than benchmark
            p95_latency_ms=100,
            p99_latency_ms=150,
            availability_percent=99.99,
        )

        plan = planner.calculate_capacity_plan(benchmark, traffic, tight_slo)

        assert not plan.meets_slo
        assert len(plan.slo_violations) > 0

    def test_estimate_replicas_for_rps(self, planner, benchmark):
        """Test replica estimation for RPS target."""
        replicas = planner.estimate_replicas_for_rps(
            benchmark=benchmark,
            target_rps=30,
            headroom_percent=20,
        )

        # 30 RPS with 20% headroom = 36 RPS
        # benchmark.rps = 15, so need ceil(36/15) = 3 replicas
        assert replicas >= 2

    def test_estimate_gpu_memory(self, planner):
        """Test GPU memory estimation."""
        memory = planner.estimate_gpu_memory(
            model_size_billions=7,
            quantization="fp16",
            context_length=4096,
            batch_size=1,
        )

        # 7B model in FP16 should need ~14GB + overhead
        assert memory > 14
        assert memory < 30

    def test_generates_alternatives(self, planner, benchmark, traffic, slo):
        """Test that capacity plan includes alternatives."""
        plan = planner.calculate_capacity_plan(benchmark, traffic, slo)

        assert len(plan.alternatives) > 0
        for alt in plan.alternatives:
            assert alt.gpu_type != benchmark.gpu_type or alt.gpu_count != benchmark.gpu_count


class TestModelRecommender:
    """Test model recommender."""

    @pytest.fixture
    def recommender(self):
        """Create model recommender."""
        return ModelRecommender()

    def test_recommend_models_by_family(self, recommender):
        """Test recommending models by family."""
        requirements = ModelRequirements(
            model_family="llama",
            capabilities=["chat"],
        )

        recommendations = recommender.recommend_models(requirements)

        assert len(recommendations) > 0
        for rec in recommendations:
            assert rec.model_family == "llama"

    def test_recommend_models_by_capability(self, recommender):
        """Test recommending models by capability."""
        requirements = ModelRequirements(
            capabilities=["code"],
        )

        recommendations = recommender.recommend_models(requirements)

        assert len(recommendations) > 0

    def test_recommend_models_with_size_limit(self, recommender):
        """Test recommending models with size limit."""
        requirements = ModelRequirements(
            max_parameters=7,
            capabilities=["chat"],
        )

        recommendations = recommender.recommend_models(requirements)

        for rec in recommendations:
            if rec.score > 50:  # Only check passing recommendations
                assert rec.estimated_size_gb <= 14  # FP16 of 7B

    def test_recommend_gpu(self, recommender):
        """Test GPU recommendation."""
        rec = recommender.recommend_gpu(
            model_size_billions=7,
            quantization="fp16",
            target_latency_ms=500,
        )

        assert rec.gpu_type in ["T4", "L4", "A10", "A100-40GB", "A100-80GB"]
        assert rec.gpu_count >= 1

    def test_recommend_gpu_large_model(self, recommender):
        """Test GPU recommendation for large model."""
        rec = recommender.recommend_gpu(
            model_size_billions=70,
            quantization="fp16",
        )

        # 70B model needs large GPU or multiple GPUs
        assert rec.gpu_type in ["A100-80GB", "H100-80GB"]


class TestWhatIfAnalyzer:
    """Test what-if analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create what-if analyzer."""
        return WhatIfAnalyzer()

    @pytest.fixture
    def setup_data(self):
        """Create test data for what-if analysis."""
        benchmark = BenchmarkData(
            model_name="llama-2-7b",
            model_version="1.0",
            gpu_type="A100-40GB",
            gpu_count=1,
            p50_latency_ms=150,
            p95_latency_ms=250,
            p99_latency_ms=400,
            tokens_per_second=120,
            requests_per_second=15,
            gpu_memory_gb=14,
            gpu_utilization_percent=85,
        )

        traffic = TrafficProfile(
            pattern=TrafficPattern.STEADY,
            requests_per_second=30,
        )

        slo = SLORequirements(
            p50_latency_ms=500,
            p95_latency_ms=1000,
            p99_latency_ms=2000,
        )

        planner = CapacityPlanner()
        original_plan = planner.calculate_capacity_plan(benchmark, traffic, slo)

        return benchmark, traffic, slo, original_plan

    def test_analyze_traffic_growth(self, analyzer, setup_data):
        """Test analyzing traffic growth scenarios."""
        benchmark, traffic, slo, original_plan = setup_data

        scenario = WhatIfScenario(
            name="Double traffic",
            rps_multiplier=2.0,
        )

        result = analyzer.analyze_scenario(
            original_plan=original_plan,
            scenario=scenario,
            benchmark=benchmark,
            traffic_profile=traffic,
            slo_requirements=slo,
        )

        assert result.replica_delta > 0  # Should need more replicas

    def test_analyze_slo_tightening(self, analyzer, setup_data):
        """Test analyzing tighter SLO requirements."""
        benchmark, traffic, slo, original_plan = setup_data

        scenario = WhatIfScenario(
            name="Tighter latency",
            p95_latency_ms=500,  # Tighter than original 1000ms
        )

        result = analyzer.analyze_scenario(
            original_plan=original_plan,
            scenario=scenario,
            benchmark=benchmark,
            traffic_profile=traffic,
            slo_requirements=slo,
        )

        # Result should include assessment
        assert result.latency_delta_percent != 0

    def test_analyze_gpu_switch(self, analyzer, setup_data):
        """Test analyzing GPU type change."""
        benchmark, traffic, slo, original_plan = setup_data

        scenario = WhatIfScenario(
            name="Switch to A100-80GB",
            gpu_type="A100-80GB",
        )

        result = analyzer.analyze_scenario(
            original_plan=original_plan,
            scenario=scenario,
            benchmark=benchmark,
            traffic_profile=traffic,
            slo_requirements=slo,
        )

        assert result.modified_plan.gpu_type == "A100-80GB"
