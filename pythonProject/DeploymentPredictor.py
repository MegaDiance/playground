import pandas as pd
import numpy as np
from deployment_predictor import DeploymentPredictor

# 1. First, prepare your historical deployment data
historical_deployments = [
    {
        'changes': {
            'config_type': 'database',
            'component': 'connection-pool',
            'new_value': 'max_connections=100;min_connections=10;idle_timeout=300',
            'is_critical': True,
            'environment': 'prod',
            'dependencies': ['cache-service', 'user-service'],
            'deployment_time': '14:00',
            'day_of_week': 'Monday'
        },
        'success': True
    },
    {
        'changes': {
            'config_type': 'application',
            'component': 'thread-pool',
            'new_value': 'max_threads=50;core_threads=20',
            'is_critical': True,
            'environment': 'prod',
            'dependencies': ['logging-service'],
            'deployment_time': '15:00',
            'day_of_week': 'Wednesday'
        },
        'success': True
    },
    {
        'changes': {
            'config_type': 'cache',
            'component': 'redis-config',
            'new_value': 'maxmemory=4gb;maxmemory-policy=allkeys-lru',
            'is_critical': True,
            'environment': 'prod',
            'dependencies': [],
            'deployment_time': '23:00',
            'day_of_week': 'Friday'
        },
        'success': False  # Failed deployment
    },
    {
        'changes': {
            'config_type': 'security',
            'component': 'ssl-config',
            'new_value': 'ssl.enabled=true;ssl.protocol=TLS1.2',
            'is_critical': True,
            'environment': 'prod',
            'dependencies': ['auth-service', 'gateway'],
            'deployment_time': '02:00',
            'day_of_week': 'Saturday'
        },
        'success': False  # Failed deployment
    }
]


def collect_deployment_data_from_logs(start_date, end_date):
    """
    Example function to collect deployment data from your logs/monitoring system
    Replace this with your actual log collection logic
    """
    # This is a placeholder - implement your actual log collection logic here
    # Example structure:
    deployment_data = []

    # You might want to query your logging system, CI/CD pipeline, or deployment platform
    # Example using hypothetical logging system:
    """
    logs = deployment_logging_system.query(
        start_date=start_date,
        end_date=end_date,
        event_type='deployment'
    )

    for log in logs:
        deployment_data.append({
            'changes': {
                'config_type': log.config_type,
                'component': log.component,
                'new_value': log.configuration_changes,
                'is_critical': log.is_critical,
                'environment': log.environment,
                'dependencies': log.affected_services,
                'deployment_time': log.timestamp.strftime('%H:%M'),
                'day_of_week': log.timestamp.strftime('%A')
            },
            'success': log.deployment_status == 'SUCCESS'
        })
    """
    return deployment_data


def train_model():
    # 1. Initialize the predictor
    predictor = DeploymentPredictor()

    # 2. Collect historical data
    # Option 1: Use manually collected data
    training_data = historical_deployments

    # Option 2: Collect data from logs
    # start_date = datetime.now() - timedelta(days=90)  # Last 90 days
    # end_date = datetime.now()
    # training_data = collect_deployment_data_from_logs(start_date, end_date)

    # 3. Train the model and get feature importance
    feature_importance = predictor.fit(training_data)

    # 4. Print training insights
    print("\nModel Training Complete!")
    print("\nFeature Importance:")
    print(feature_importance)

    # 5. Validate the model with a test prediction
    test_changes = {
        'config_type': 'database',
        'component': 'connection-pool',
        'new_value': 'max_connections=200;min_connections=20',
        'is_critical': True,
        'environment': 'prod',
        'dependencies': ['cache-service'],
        'deployment_time': '14:00',
        'day_of_week': 'Tuesday'
    }

    prediction = predictor.predict_probability(test_changes)

    print("\nTest Prediction:")
    print(f"Deployment Success Probability: {prediction['success_probability']:.2%}")
    print(f"Risk Factors: {prediction['risk_factors']}")

    # 6. Save the trained model
    predictor.save_model('trained_deployment_predictor.joblib')
    print("\nModel saved to 'trained_deployment_predictor.joblib'")

    return predictor


def evaluate_model_performance(predictor, test_data):
    """
    Evaluate model performance using test data
    """
    successes = []
    predictions = []

    for deployment in test_data:
        prediction = predictor.predict_probability(deployment['changes'])
        predictions.append(prediction['success_probability'] >= 0.5)
        successes.append(deployment['success'])

    # Calculate accuracy
    accuracy = sum(1 for p, s in zip(predictions, successes) if p == s) / len(predictions)

    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.2%}")

    # You could add more metrics like precision, recall, F1 score, etc.


if __name__ == "__main__":
    # Train the model
    predictor = train_model()

    # Evaluate the model
    test_data = [
        {
            'changes': {
                'config_type': 'messaging',
                'component': 'kafka-config',
                'new_value': 'num.partitions=10;replication.factor=3',
                'is_critical': True,
                'environment': 'prod',
                'dependencies': ['messaging-service'],
                'deployment_time': '16:00',
                'day_of_week': 'Wednesday'
            },
            'success': True
        },
        # Add more test cases...
    ]

    evaluate_model_performance(predictor, test_data)