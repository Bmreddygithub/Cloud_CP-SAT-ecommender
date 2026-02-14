"""
CP-SAT Constraint Satisfaction Model for Cloud Provider Recommendation
Uses Google OR-Tools CP-SAT Solver
"""

from ortools.sat.python import cp_model
import pandas as pd
from typing import Dict, List, Optional


class CloudProviderRecommender:
    def __init__(self, data_file: str):
        """Initialize the recommender with cloud provider data."""
        self.df = pd.read_excel(data_file)
        self.providers = self.df['provider'].tolist()
        self.n_providers = len(self.providers)
        
        # Preprocess data for constraint modeling
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Convert categorical data to boolean/integer representations."""
        # Create boolean mappings for key features
        self.features = {
            'tpu_support': self.df['tpu_support'].apply(lambda x: 1 if 'Yes' in str(x) else 0).tolist(),
            'automl': self.df['automl'].apply(lambda x: 0 if pd.isna(x) or x == 'No' else 1).tolist(),
            'kubernetes_support': self.df['kubernetes_support'].apply(lambda x: 0 if pd.isna(x) or x == 'No' else 1).tolist(),
            'serverless_inference': self.df['serverless_inference'].apply(lambda x: 0 if pd.isna(x) or x == 'No' else 1).tolist(),
            'free_tier': self.df['free_tier_or_trial'].apply(lambda x: 0 if pd.isna(x) or str(x) == 'Unknown' else 1).tolist(),
            'managed_notebooks': self.df['managed_notebooks'].apply(lambda x: 0 if pd.isna(x) or x == 'No' or x == 'Unknown' else 1).tolist(),
            'spot_instances': self.df['spot_instances'].apply(lambda x: 0 if pd.isna(x) or x == 'No' or 'Unknown' in str(x) else 1).tolist(),
            'autoscaling': self.df['autoscaling'].apply(lambda x: 0 if pd.isna(x) or x == 'No' else 1).tolist(),
            'distributed_training': self.df['distributed_training'].apply(lambda x: 0 if pd.isna(x) or x == 'No' else 1).tolist(),
            'multi_cloud': self.df['multi_cloud_support'].apply(lambda x: 0 if pd.isna(x) or x == 'No' else 1).tolist(),
        }
        
        # Category encoding
        self.categories = {
            'hyperscaler': self.df['category'].apply(lambda x: 1 if x == 'Hyperscaler' else 0).tolist(),
            'gpu_cloud': self.df['category'].apply(lambda x: 1 if 'GPU' in str(x) else 0).tolist(),
            'ai_platform': self.df['category'].apply(lambda x: 1 if 'AI Platform' in str(x) else 0).tolist(),
        }
        
        # GPU types (A100, H100 availability)
        self.gpu_types = {
            'has_a100': self.df['compute_gpus'].apply(lambda x: 1 if 'A100' in str(x) else 0).tolist(),
            'has_h100': self.df['compute_gpus'].apply(lambda x: 1 if 'H100' in str(x) else 0).tolist(),
            'has_v100': self.df['compute_gpus'].apply(lambda x: 1 if 'V100' in str(x) else 0).tolist(),
        }
        
        # Pricing models
        self.pricing = {
            'has_spot': self.df['pricing_models'].apply(lambda x: 1 if 'Spot' in str(x) else 0).tolist(),
            'has_reserved': self.df['pricing_models'].apply(lambda x: 1 if 'Reserved' in str(x) or 'Committed' in str(x) else 0).tolist(),
        }
        
        # Regional scope
        self.regions = {
            'is_global': self.df['region_scope'].apply(lambda x: 1 if 'Global' in str(x) else 0).tolist(),
            'is_regional': self.df['region_scope'].apply(lambda x: 1 if 'Regional' in str(x) or 'region' in str(x).lower() else 0).tolist(),
        }
        
    def find_recommendations(self, constraints: Dict, max_solutions: int = 5) -> List[Dict]:
        """
        Find cloud providers that satisfy all constraints.
        
        Args:
            constraints: Dictionary of constraint specifications
            max_solutions: Maximum number of solutions to find
            
        Returns:
            List of recommended providers with details
        """
        model = cp_model.CpModel()
        
        # Create boolean variables for each provider
        provider_vars = []
        for i in range(self.n_providers):
            provider_vars.append(model.NewBoolVar(f'provider_{i}'))
        
        # Apply constraints based on user requirements
        
        # 1. Feature constraints (MANDATORY if specified)
        if constraints.get('requires_tpu', False):
            for i in range(self.n_providers):
                if self.features['tpu_support'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        if constraints.get('requires_automl', False):
            for i in range(self.n_providers):
                if self.features['automl'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        if constraints.get('requires_kubernetes', False):
            for i in range(self.n_providers):
                if self.features['kubernetes_support'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        if constraints.get('requires_serverless', False):
            for i in range(self.n_providers):
                if self.features['serverless_inference'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        if constraints.get('needs_free_tier', False):
            for i in range(self.n_providers):
                if self.features['free_tier'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        if constraints.get('requires_managed_notebooks', False):
            for i in range(self.n_providers):
                if self.features['managed_notebooks'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        if constraints.get('requires_spot_instances', False):
            for i in range(self.n_providers):
                if self.features['spot_instances'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        if constraints.get('requires_autoscaling', False):
            for i in range(self.n_providers):
                if self.features['autoscaling'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        if constraints.get('requires_distributed_training', False):
            for i in range(self.n_providers):
                if self.features['distributed_training'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        if constraints.get('requires_multi_cloud', False):
            for i in range(self.n_providers):
                if self.features['multi_cloud'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        # 2. Category constraints
        category = constraints.get('category', None)
        if category == 'hyperscaler':
            for i in range(self.n_providers):
                if self.categories['hyperscaler'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        elif category == 'gpu_cloud':
            for i in range(self.n_providers):
                if self.categories['gpu_cloud'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        # 3. GPU type constraints
        gpu_requirement = constraints.get('gpu_type', None)
        if gpu_requirement == 'A100':
            for i in range(self.n_providers):
                if self.gpu_types['has_a100'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        elif gpu_requirement == 'H100':
            for i in range(self.n_providers):
                if self.gpu_types['has_h100'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        elif gpu_requirement == 'V100':
            for i in range(self.n_providers):
                if self.gpu_types['has_v100'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        # 4. Regional constraints
        if constraints.get('must_be_global', False):
            for i in range(self.n_providers):
                if self.regions['is_global'][i] == 0:
                    model.Add(provider_vars[i] == 0)
        
        # 5. At least one provider must be selected
        model.Add(sum(provider_vars) >= 1)
        
        # 6. Limit number of recommendations
        model.Add(sum(provider_vars) <= max_solutions)
        
        # Create solver and collect solutions
        solver = cp_model.CpSolver()
        solution_collector = SolutionCollector(provider_vars, self.providers, self.df, max_solutions)
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.max_time_in_seconds = 10.0
        
        status = solver.Solve(model, solution_collector)
        
        return solution_collector.get_recommendations()


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    """Callback to collect all solutions."""
    
    def __init__(self, variables, provider_names, df, max_solutions):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._variables = variables
        self._provider_names = provider_names
        self._df = df
        self._max_solutions = max_solutions
        self._solutions = []
        self._solution_count = 0
    
    def on_solution_callback(self):
        """Called for each solution found."""
        if self._solution_count >= self._max_solutions:
            self.StopSearch()
            return
        
        selected_providers = []
        for i, var in enumerate(self._variables):
            if self.Value(var) == 1:
                provider_info = {
                    'provider': self._provider_names[i],
                    'category': self._df.iloc[i]['category'],
                    'primary_ml_service': self._df.iloc[i]['primary_ml_service'],
                    'compute_gpus': self._df.iloc[i]['compute_gpus'],
                    'tpu_support': self._df.iloc[i]['tpu_support'],
                    'pricing_models': self._df.iloc[i]['pricing_models'],
                    'automl': self._df.iloc[i]['automl'],
                    'kubernetes_support': self._df.iloc[i]['kubernetes_support'],
                    'managed_notebooks': self._df.iloc[i]['managed_notebooks'],
                    'serverless_inference': self._df.iloc[i]['serverless_inference'],
                    'free_tier_or_trial': self._df.iloc[i]['free_tier_or_trial'],
                    'notes': self._df.iloc[i]['notes'],
                }
                selected_providers.append(provider_info)
        
        if selected_providers:
            self._solutions.append(selected_providers)
            self._solution_count += 1
    
    def get_recommendations(self):
        """Return all collected solutions."""
        return self._solutions


def run_example_scenarios():
    """Run multiple example scenarios demonstrating CP-SAT recommendations."""
    
    print("=" * 80)
    print("CP-SAT Cloud Provider Recommender - Example Scenarios")
    print("=" * 80)
    
    recommender = CloudProviderRecommender('cloud_providers_ml_recommender_dataset_v1.xlsx')
    
    # Scenario 1: Startup needs free tier with AutoML
    print("\n\n" + "="*80)
    print("SCENARIO 1: Startup looking for free tier with AutoML capabilities")
    print("="*80)
    constraints1 = {
        'needs_free_tier': True,
        'requires_automl': True,
        'requires_managed_notebooks': True,
    }
    print(f"\nConstraints: {constraints1}")
    solutions1 = recommender.find_recommendations(constraints1, max_solutions=3)
    print_solutions(solutions1, "Scenario 1")
    
    # Scenario 2: Enterprise needs H100 GPUs with Kubernetes
    print("\n\n" + "="*80)
    print("SCENARIO 2: Enterprise ML team needs H100 GPUs with Kubernetes support")
    print("="*80)
    constraints2 = {
        'gpu_type': 'H100',
        'requires_kubernetes': True,
        'requires_distributed_training': True,
        'requires_spot_instances': True,
    }
    print(f"\nConstraints: {constraints2}")
    solutions2 = recommender.find_recommendations(constraints2, max_solutions=5)
    print_solutions(solutions2, "Scenario 2")
    
    # Scenario 3: Research lab needs TPUs with serverless
    print("\n\n" + "="*80)
    print("SCENARIO 3: Research lab needs TPU support with serverless inference")
    print("="*80)
    constraints3 = {
        'requires_tpu': True,
        'requires_serverless': True,
        'requires_autoscaling': True,
    }
    print(f"\nConstraints: {constraints3}")
    solutions3 = recommender.find_recommendations(constraints3, max_solutions=3)
    print_solutions(solutions3, "Scenario 3")
    
    # Scenario 4: Multi-cloud deployment
    print("\n\n" + "="*80)
    print("SCENARIO 4: Company needs multi-cloud support with A100 GPUs")
    print("="*80)
    constraints4 = {
        'gpu_type': 'A100',
        'requires_multi_cloud': True,
        'requires_kubernetes': True,
    }
    print(f"\nConstraints: {constraints4}")
    solutions4 = recommender.find_recommendations(constraints4, max_solutions=3)
    print_solutions(solutions4, "Scenario 4")
    
    # Scenario 5: Hyperscaler with all features
    print("\n\n" + "="*80)
    print("SCENARIO 5: Looking for hyperscaler with comprehensive ML features")
    print("="*80)
    constraints5 = {
        'category': 'hyperscaler',
        'requires_automl': True,
        'requires_managed_notebooks': True,
        'requires_serverless': True,
        'must_be_global': True,
    }
    print(f"\nConstraints: {constraints5}")
    solutions5 = recommender.find_recommendations(constraints5, max_solutions=3)
    print_solutions(solutions5, "Scenario 5")


def print_solutions(solutions, scenario_name):
    """Pretty print solutions."""
    if not solutions:
        print("\n❌ No providers found satisfying all constraints!")
        return
    
    print(f"\n✅ Found {len(solutions)} solution(s):")
    
    for sol_idx, solution in enumerate(solutions, 1):
        print(f"\n{'─'*80}")
        print(f"Solution #{sol_idx}: {len(solution)} provider(s)")
        print(f"{'─'*80}")
        
        for provider_info in solution:
            print(f"\n🔹 Provider: {provider_info['provider']}")
            print(f"   Category: {provider_info['category']}")
            print(f"   ML Service: {provider_info['primary_ml_service']}")
            print(f"   GPUs: {provider_info['compute_gpus']}")
            print(f"   TPU Support: {provider_info['tpu_support']}")
            print(f"   Pricing: {provider_info['pricing_models']}")
            print(f"   AutoML: {provider_info['automl']}")
            print(f"   Kubernetes: {provider_info['kubernetes_support']}")
            print(f"   Managed Notebooks: {provider_info['managed_notebooks']}")
            print(f"   Serverless: {provider_info['serverless_inference']}")
            print(f"   Free Tier: {provider_info['free_tier_or_trial']}")
            print(f"   Notes: {provider_info['notes']}")


if __name__ == "__main__":
    # Install ortools if needed: pip install ortools
    run_example_scenarios()
