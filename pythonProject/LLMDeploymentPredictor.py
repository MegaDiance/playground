from typing import Dict, List, Any
import openai
import json
from datetime import datetime
import os


class LLMDeploymentPredictor:
    def __init__(self, api_key: str):
        """Initialize the LLM-based deployment predictor"""
        openai.api_key = api_key

        self.system_prompt = """You are an expert DevOps engineer specializing in deployment risk assessment. 
        Analyze the given deployment changes and predict the probability of successful deployment.
        Consider factors like:
        1. Configuration complexity
        2. Dependencies
        3. Deployment timing
        4. Historical patterns
        5. Critical service impacts

        Provide:
        1. Success probability (0-100%)
        2. Specific risk factors
        3. Recommendations for risk mitigation

        Format your response as JSON with keys: probability, risk_factors, recommendations"""

    def analyze_deployment(self, changes: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze deployment changes using LLM

        Args:
            changes: Dictionary containing deployment changes
            context: Historical context and additional information
        """
        # Construct the prompt
        prompt = self._construct_prompt(changes, context)

        try:
            # Get LLM prediction
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent predictions
                max_tokens=1000
            )

            # Parse response
            result = json.loads(response.choices[0].message.content)

            # Add metadata
            result['timestamp'] = datetime.now().isoformat()
            result['model_version'] = 'gpt-4'

            return result

        except Exception as e:
            raise Exception(f"Error getting LLM prediction: {str(e)}")

    def _construct_prompt(self, changes: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Construct detailed prompt for the LLM"""
        prompt = f"""Please analyze the following deployment changes and predict deployment success probability:

Configuration Changes:
Type: {changes.get('config_type')}
Component: {changes.get('component')}
Changes: {changes.get('new_value')}
Critical: {changes.get('is_critical', False)}
Environment: {changes.get('environment')}
Deployment Time: {changes.get('deployment_time')}
Day: {changes.get('day_of_week')}
Dependencies: {', '.join(changes.get('dependencies', []))}

Historical Context:
Past Deployments: {context.get('total_deployments', 0)}
Success Rate: {context.get('success_rate', 0)}%
Recent Failures: {context.get('recent_failures', 0)}
Similar Deployment Success: {context.get('similar_deployment_success', 0)}%

Additional Context:
Team Experience: {context.get('team_experience', 'medium')}
Rollback Plan: {context.get('has_rollback_plan', False)}
Testing Coverage: {context.get('testing_coverage', 0)}%
"""
        return prompt


def compare_approaches():
    """Compare LLM vs Traditional ML approaches"""

    # Example deployment change
    changes = {
        'config_type': 'database',
        'component': 'connection-pool',
        'new_value': 'max_connections=200;min_connections=20;idle_timeout=300',
        'is_critical': True,
        'environment': 'prod',
        'dependencies': ['cache-service', 'user-service'],
        'deployment_time': '14:00',
        'day_of_week': 'Monday'
    }

    # Historical context
    context = {
        'total_deployments': 150,
        'success_rate': 85,
        'recent_failures': 2,
        'similar_deployment_success': 78,
        'team_experience': 'high',
        'has_rollback_plan': True,
        'testing_coverage': 80
    }

    # Initialize LLM predictor
    llm_predictor = LLMDeploymentPredictor(api_key=os.getenv('OPENAI_API_KEY'))

    # Get LLM prediction
    llm_result = llm_predictor.analyze_deployment(changes, context)

    return llm_result


# Example usage
if __name__ == "__main__":
    result = compare_approaches()
    print("\nLLM Prediction:")
    print(f"Success Probability: {result['probability']}%")
    print("\nRisk Factors:")
    for factor in result['risk_factors']:
        print(f"- {factor}")
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"- {rec}")