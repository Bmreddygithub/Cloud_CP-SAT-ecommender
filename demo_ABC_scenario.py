"""
Demo Test Case: ABC Enterprise GPU Provider Selection
Real-world scenario for cost-effective long-term GPU cloud selection
"""

from cloud_cpsat_recommender import CloudProviderRecommender


def demo_ABC_cost_effective_gpu_scenario():
    """
    DEMO SCENARIO: ABC Needs Cost-Effective Long-Term GPU Provider
    
    Company Profile:
    - Company: ABC (Large telecommunications and media conglomerate)
    - Use Case: AI/ML workloads for video streaming optimization, 
                content recommendation, network traffic prediction
    - Scale: Enterprise-level (thousands of models, 24/7 operations)
    - Duration: Long-term commitment (3-5 years)
    
    Business Requirements:
    1. Cost Optimization:
       - Reserved/committed pricing for predictable costs
       - Spot/preemptible instances for batch workloads
       - Autoscaling to optimize resource utilization
    
    2. GPU Requirements:
       - Modern GPUs (A100 or H100) for training large models
       - High availability for production inference
    
    3. Enterprise Features:
       - Kubernetes support (existing infrastructure)
       - Multi-cloud capability (avoid vendor lock-in)
       - Global presence (serve customers worldwide)
       - Managed infrastructure (reduce ops overhead)
    
    4. MLOps Requirements:
       - Serverless inference for cost efficiency
       - Managed notebooks for data science team
       - Distributed training for large models
    """
    
    print("=" * 80)
    print("DEMO: ABC ENTERPRISE GPU PROVIDER SELECTION")
    print("=" * 80)
    print("\n📊 BUSINESS CONTEXT:")
    print("   Company: ABC Corporation")
    print("   Industry: Telecommunications & Media")
    print("   Team Size: 50+ data scientists & ML engineers")
    print("   Budget: $2-5M annually for ML infrastructure")
    print("   Timeline: Long-term (3-5 year commitment)")
    
    print("\n🎯 OBJECTIVES:")
    print("   1. Minimize long-term costs through reserved pricing")
    print("   2. Optimize variable costs with spot instances")
    print("   3. Maintain enterprise-grade reliability and security")
    print("   4. Enable data science team productivity")
    print("   5. Avoid vendor lock-in with multi-cloud strategy")
    
    print("\n" + "-" * 80)
    
    recommender = CloudProviderRecommender('cloud_providers_ml_recommender_dataset_v1.xlsx')
    
    # Test Case 1: Strict Cost Optimization (Most Constrained)
    print("\n\n" + "=" * 80)
    print("TEST CASE 1: Maximum Cost Optimization")
    print("Priority: COST > Features")
    print("=" * 80)
    
    constraints_cost_focused = {
        'gpu_type': 'A100',  # Mature, widely available, good price/performance
        'requires_spot_instances': True,  # Cost savings on batch jobs
        'requires_kubernetes': True,  # Existing infrastructure
        'requires_autoscaling': True,  # Optimize resource utilization
        'requires_distributed_training': True,  # Training efficiency
    }
    
    print("\n📋 Constraints:")
    for key, value in constraints_cost_focused.items():
        print(f"   • {key}: {value}")
    
    solutions_1 = recommender.find_recommendations(constraints_cost_focused, max_solutions=5)
    
    print(f"\n✅ Found {len(solutions_1)} solution(s):\n")
    
    if solutions_1:
        for sol_idx, solution in enumerate(solutions_1[:3], 1):  # Show top 3
            print(f"{'─' * 80}")
            print(f"Solution #{sol_idx}: {len(solution)} provider(s)")
            print(f"{'─' * 80}")
            
            for provider in solution:
                print(f"\n🔹 {provider['provider']}")
                print(f"   Category: {provider['category']}")
                print(f"   GPUs: {provider['compute_gpus']}")
                print(f"   Pricing Models: {provider['pricing_models']}")
                print(f"   Kubernetes: {provider['kubernetes_support']}")
                print(f"   Spot Instances: Available ✓")
                print(f"   Autoscaling: Available ✓")
                print(f"   💡 Cost Advantage: {get_cost_benefit(provider)}")
                print(f"   📊 Best For: {get_use_case(provider)}")
    else:
        print("❌ No providers found matching all constraints")
    
    # Test Case 2: Enterprise Features + Cost Balance
    print("\n\n" + "=" * 80)
    print("TEST CASE 2: Enterprise Features + Cost Balance")
    print("Priority: COST = Features (Balanced)")
    print("=" * 80)
    
    constraints_balanced = {
        'gpu_type': 'H100',  # Latest performance for future-proofing
        'requires_kubernetes': True,  # Infrastructure requirement
        'requires_spot_instances': True,  # Cost optimization
        'requires_serverless': True,  # Variable workload efficiency
        'requires_managed_notebooks': True,  # Team productivity
        'requires_distributed_training': True,  # Large model training
        'must_be_global': True,  # Global operations
    }
    
    print("\n📋 Constraints:")
    for key, value in constraints_balanced.items():
        print(f"   • {key}: {value}")
    
    solutions_2 = recommender.find_recommendations(constraints_balanced, max_solutions=5)
    
    print(f"\n✅ Found {len(solutions_2)} solution(s):\n")
    
    if solutions_2:
        for sol_idx, solution in enumerate(solutions_2[:2], 1):  # Show top 2
            print(f"{'─' * 80}")
            print(f"Solution #{sol_idx}: {len(solution)} provider(s)")
            print(f"{'─' * 80}")
            
            total_cost_score = 0
            for provider in solution:
                cost_score = calculate_cost_score(provider)
                feature_score = calculate_feature_score(provider)
                total_cost_score += cost_score
                
                print(f"\n🔹 {provider['provider']}")
                print(f"   Category: {provider['category']}")
                print(f"   ML Service: {provider['primary_ml_service']}")
                print(f"   GPUs: {provider['compute_gpus']}")
                print(f"   Pricing: {provider['pricing_models']}")
                print(f"   Kubernetes: {provider['kubernetes_support']}")
                print(f"   Serverless: {provider['serverless_inference']}")
                print(f"   Notebooks: {provider['managed_notebooks']}")
                print(f"   💰 Cost Score: {cost_score}/10")
                print(f"   ⭐ Feature Score: {feature_score}/10")
                print(f"   📈 Total Value: {cost_score + feature_score}/20")
    else:
        print("❌ No providers found matching all constraints")
    
    # Test Case 3: Multi-Cloud Strategy
    print("\n\n" + "=" * 80)
    print("TEST CASE 3: Multi-Cloud Strategy (Avoid Vendor Lock-in)")
    print("Priority: Flexibility > Cost")
    print("=" * 80)
    
    constraints_multicloud = {
        'gpu_type': 'A100',
        'requires_multi_cloud': True,  # Critical for avoiding lock-in
        'requires_kubernetes': True,
        'requires_serverless': True,
        'requires_autoscaling': True,
    }
    
    print("\n📋 Constraints:")
    for key, value in constraints_multicloud.items():
        print(f"   • {key}: {value}")
    
    solutions_3 = recommender.find_recommendations(constraints_multicloud, max_solutions=5)
    
    print(f"\n✅ Found {len(solutions_3)} solution(s):\n")
    
    if solutions_3:
        for sol_idx, solution in enumerate(solutions_3[:3], 1):
            print(f"{'─' * 80}")
            print(f"Solution #{sol_idx}: {len(solution)} provider(s)")
            print(f"{'─' * 80}")
            
            for provider in solution:
                print(f"\n🔹 {provider['provider']}")
                print(f"   Category: {provider['category']}")
                print(f"   GPUs: {provider['compute_gpus']}")
                print(f"   Multi-Cloud: Yes ✓")
                print(f"   Kubernetes: {provider['kubernetes_support']}")
                print(f"   🌍 Strategy: {get_multicloud_strategy(provider)}")
    else:
        print("❌ No providers found matching all constraints")
    
    # Final Recommendation Summary
    print("\n\n" + "=" * 80)
    print("FINAL RECOMMENDATION SUMMARY FOR ABC")
    print("=" * 80)
    
    print("\n🏆 RECOMMENDED APPROACH:")
    print("\n1. PRIMARY PROVIDER (70% workload):")
    print("   → Google Cloud or Microsoft Azure")
    print("   → Reason: Best balance of cost, features, and enterprise support")
    print("   → Use committed/reserved pricing for predictable workloads")
    print("   → H100 GPUs for training, A100 for inference")
    
    print("\n2. SECONDARY PROVIDER (20% workload):")
    print("   → RunPod or CoreWeave")
    print("   → Reason: Cost-effective for burst capacity and batch jobs")
    print("   → Use spot instances for non-critical workloads")
    print("   → Lower cost per GPU hour for experimentation")
    
    print("\n3. BACKUP/FAILOVER (10% capacity):")
    print("   → Alternative hyperscaler (Azure if primary is GCP, or vice versa)")
    print("   → Reason: Business continuity and disaster recovery")
    print("   → Maintain multi-cloud capability")
    
    print("\n💰 COST OPTIMIZATION STRATEGY:")
    print("   ✓ 60% Reserved/Committed instances (production workloads)")
    print("   ✓ 30% Spot/Preemptible instances (batch training)")
    print("   ✓ 10% On-demand (testing, variable workloads)")
    print("   📊 Estimated savings: 40-60% vs. pure on-demand")
    
    print("\n📈 EXPECTED ANNUAL COST BREAKDOWN:")
    print("   • GPU compute (reserved): $1.8M - $2.5M")
    print("   • GPU compute (spot): $400K - $600K")
    print("   • Storage & networking: $200K - $300K")
    print("   • Support & services: $150K - $250K")
    print("   • Total: $2.55M - $3.65M (within $2-5M budget)")
    
    print("\n🎯 KEY SUCCESS METRICS:")
    print("   • Cost per training job: Tracked monthly")
    print("   • GPU utilization: Target >75%")
    print("   • Spot instance interruption rate: <5%")
    print("   • Model deployment latency: <100ms p99")
    print("   • Total infrastructure cost as % of ML budget: <40%")
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE: Decision-ready recommendation generated")
    print("=" * 80)


def get_cost_benefit(provider):
    """Return cost benefit description for provider."""
    provider_name = provider['provider']
    
    cost_benefits = {
        'Google Cloud': 'Sustained use discounts + committed use up to 55% off',
        'Microsoft Azure': 'Reserved instances up to 72% off, spot up to 90% off',
        'AWS': 'Reserved instances + savings plans, spot up to 90% off',
        'RunPod': 'Lowest per-hour GPU cost, instant scaling',
        'CoreWeave': 'HPC-optimized, competitive hourly rates',
        'Lambda Labs': 'Simple pricing, no hidden fees',
    }
    
    return cost_benefits.get(provider_name, 'Competitive pricing with spot instances')


def get_use_case(provider):
    """Return best use case for provider."""
    provider_name = provider['provider']
    
    use_cases = {
        'Google Cloud': 'Production ML pipelines, TPU workloads',
        'Microsoft Azure': 'Enterprise integration, Windows workloads',
        'AWS': 'Comprehensive ecosystem, mature services',
        'RunPod': 'Burst capacity, cost-sensitive batch jobs',
        'CoreWeave': 'High-performance training, rendering',
        'Lambda Labs': 'Research, experimentation, training',
    }
    
    return use_cases.get(provider_name, 'General ML workloads')


def get_multicloud_strategy(provider):
    """Return multi-cloud strategy for provider."""
    provider_name = provider['provider']
    
    strategies = {
        'Microsoft Azure': 'Anthos/AKS for cross-cloud Kubernetes',
        'Google Cloud': 'Anthos for hybrid/multi-cloud',
        'AWS': 'EKS Anywhere for multi-cloud K8s',
    }
    
    return strategies.get(provider_name, 'Container-based portability')


def calculate_cost_score(provider):
    """Calculate cost effectiveness score (0-10)."""
    score = 5  # Base score
    
    pricing = str(provider['pricing_models']).lower()
    if 'spot' in pricing or 'preemptible' in pricing:
        score += 2
    if 'reserved' in pricing or 'committed' in pricing:
        score += 2
    
    if provider['category'] == 'Hyperscaler':
        score += 1  # Economies of scale
    elif 'GPU Cloud' in provider['category']:
        score += 2  # Often more cost-effective
    
    return min(score, 10)


def calculate_feature_score(provider):
    """Calculate feature richness score (0-10)."""
    score = 0
    
    if provider.get('automl') not in ['No', None]:
        score += 1
    if provider.get('kubernetes_support') not in ['No', None]:
        score += 1
    if provider.get('managed_notebooks') not in ['No', 'Unknown', None]:
        score += 1
    if provider.get('serverless_inference') not in ['No', None]:
        score += 1
    if provider.get('category') == 'Hyperscaler':
        score += 2  # Comprehensive ecosystem
    if 'H100' in str(provider.get('compute_gpus', '')):
        score += 1  # Latest hardware
    if 'Global' in str(provider.get('region_scope', '')):
        score += 1  # Global presence
    if provider.get('pricing_models'):
        pricing = str(provider['pricing_models']).lower()
        if 'spot' in pricing or 'reserved' in pricing:
            score += 1
    
    return min(score, 10)


if __name__ == "__main__":
    demo_ABC_cost_effective_gpu_scenario()
