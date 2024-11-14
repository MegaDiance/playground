import pandas as pd
import numpy as np
from github import Github
from datetime import datetime, timedelta
import yaml
import re
from typing import List, Dict, Any
import os
from deployment_predictor import DeploymentPredictor


class DeploymentDataCollector:
    def __init__(self, github_token: str):
        self.github = Github(github_token)

    def extract_config_changes(self, diff_content: str) -> List[Dict[str, Any]]:
        """
        Extract configuration changes from PR diff content
        """
        changes = []

        # Common configuration file patterns
        config_patterns = {
            'application': r'application.*\.yml$|application.*\.properties$',
            'database': r'.*database.*\.yml$|.*db.*\.properties$',
            'security': r'.*security.*\.yml$|.*ssl.*\.properties$',
            'cache': r'.*cache.*\.yml$|.*redis.*\.properties$',
            'messaging': r'.*kafka.*\.yml$|.*rabbit.*\.properties$'
        }

        # Extract changes from different file types
        for line in diff_content.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                for config_type, pattern in config_patterns.items():
                    if re.search(pattern, line):
                        changes.append({
                            'config_type': config_type,
                            'component': self._identify_component(line),
                            'new_value': line.lstrip('+').strip(),
                            'is_critical': self._is_critical_change(line)
                        })

        return changes

    def _identify_component(self, config_line: str) -> str:
        """
        Identify the component from a configuration line
        """
        component_patterns = {
            'connection-pool': r'pool|connection|datasource',
            'thread-pool': r'thread|executor',
            'cache-config': r'cache|redis',
            'security-config': r'security|ssl|cert',
            'logging-config': r'log|logging',
            'messaging-config': r'kafka|rabbit|queue'
        }

        for component, pattern in component_patterns.items():
            if re.search(pattern, config_line.lower()):
                return component

        return 'other'

    def _is_critical_change(self, config_line: str) -> bool:
        """
        Determine if a configuration change is critical
        """
        critical_patterns = [
            r'password',
            r'secret',
            r'ssl',
            r'connection',
            r'pool\.size',
            r'max',
            r'timeout',
            r'critical',
            r'mandatory'
        ]

        return any(re.search(pattern, config_line.lower()) for pattern in critical_patterns)

    def parse_runsheet(self, runsheet_path: str) -> List[Dict[str, Any]]:
        """
        Parse deployment runsheet (YAML format)
        """
        with open(runsheet_path, 'r') as file:
            runsheet = yaml.safe_load(file)

        deployments = []

        for deployment in runsheet.get('deployments', []):
            deployment_data = {
                'timestamp': deployment.get('timestamp'),
                'environment': deployment.get('environment'),
                'success': deployment.get('status') == 'success',
                'dependencies': deployment.get('dependencies', []),
                'pr_number': deployment.get('pr_number'),
                'components_affected': deployment.get('components', [])
            }
            deployments.append(deployment_data)

        return deployments

    def collect_training_data(self, repo_name: str, runsheet_path: str,
                              start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Collect training data from both GitHub PRs and deployment runsheet
        """
        # Get repository
        repo = self.github.get_repo(repo_name)

        # Get all PRs in date range
        pulls = repo.get_pulls(state='closed', sort='updated', direction='desc')

        # Parse runsheet
        runsheet_deployments = self.parse_runsheet(runsheet_path)

        training_data = []

        # Match PRs with runsheet deployments and extract training data
        for pr in pulls:
            if start_date <= pr.closed_at <= end_date:
                # Find matching deployment in runsheet
                deployment = next(
                    (d for d in runsheet_deployments if d['pr_number'] == pr.number),
                    None
                )

                if deployment:
                    # Extract config changes from PR
                    config_changes = self.extract_config_changes(pr.get_files().get_page(0)[0].patch)

                    # Create training record
                    for change in config_changes:
                        training_record = {
                            'changes': {
                                'config_type': change['config_type'],
                                'component': change['component'],
                                'new_value': change['new_value'],
                                'is_critical': change['is_critical'],
                                'environment': deployment['environment'],
                                'dependencies': deployment['dependencies'],
                                'deployment_time': deployment['timestamp'].strftime('%H:%M'),
                                'day_of_week': deployment['timestamp'].strftime('%A')
                            },
                            'success': deployment['success']
                        }
                        training_data.append(training_record)

        return training_data


def train_model_from_github():
    # Initialize collector
    github_token = os.getenv('GITHUB_TOKEN')
    collector = DeploymentDataCollector(github_token)

    # Collect training data
    training_data = collector.collect_training_data(
        repo_name='your-org/your-repo',
        runsheet_path='path/to/deployment_runsheet.yml',
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now()
    )

    # Initialize and train predictor
    predictor = DeploymentPredictor()
    feature_importance = predictor.fit(training_data)

    print("\nModel Training Complete!")
    print("\nFeature Importance:")
    print(feature_importance)

    # Save the model
    predictor.save_model('github_trained_predictor.joblib')

    return predictor


# Example deployment runsheet format (YAML)
example_runsheet = """
deployments:
  - timestamp: "2024-01-15T14:30:00Z"
    environment: prod
    status: success
    pr_number: 123
    components: 
      - database
      - cache
    dependencies:
      - user-service
      - auth-service

  - timestamp: "2024-01-16T16:45:00Z"
    environment: prod
    status: failed
    pr_number: 124
    components:
      - security
    dependencies:
      - gateway
"""

if __name__ == "__main__":
    # Example usage
    predictor = train_model_from_github()

    # Test prediction
    new_changes = {
        'config_type': 'database',
        'component': 'connection-pool',
        'new_value': 'max_connections=200',
        'is_critical': True,
        'environment': 'prod',
        'dependencies': ['cache-service'],
        'deployment_time': '14:00',
        'day_of_week': 'Tuesday'
    }

    prediction = predictor.predict_probability(new_changes)
    print(f"\nPrediction for new changes:")
    print(f"Success Probability: {prediction['success_probability']:.2%}")
    print(f"Risk Factors: {prediction['risk_factors']}")