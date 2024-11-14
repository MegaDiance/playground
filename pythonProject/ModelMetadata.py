import pymongo
import gridfs
import pickle
import redis
import joblib
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from sqlalchemy import create_engine, Column, String, LargeBinary, DateTime, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class ModelMetadata(Base):
    """SQLAlchemy model for storing model metadata"""
    __tablename__ = 'model_metadata'

    id = Column(String(50), primary_key=True)
    version = Column(String(20))
    created_at = Column(DateTime)
    accuracy = Column(Float)
    feature_importance = Column(String)  # JSON string
    training_size = Column(Integer)
    framework = Column(String(50))


class ModelStorageSystem:
    """
    A system for storing ML models in different types of databases.
    Supports MongoDB, Redis, and PostgreSQL with different trade-offs:

    MongoDB (Default/Recommended):
    + Good for large models (GridFS)
    + Flexible schema for metadata
    + Good query capabilities
    + Built-in versioning support
    - Slightly slower than Redis

    Redis:
    + Fastest access time
    + Good for real-time predictions
    - Size limitations
    - Less querying capabilities

    PostgreSQL:
    + ACID compliance
    + Good for strict schema requirements
    + Complex querying capabilities
    - Slower than MongoDB/Redis for large objects
    """

    def __init__(self, storage_type: str = 'mongodb'):
        self.storage_type = storage_type
        self.logger = logging.getLogger(__name__)

        if storage_type == 'mongodb':
            # MongoDB setup (recommended for ML models)
            self.mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
            self.db = self.mongo_client.ml_models
            self.fs = gridfs.GridFS(self.db)

        elif storage_type == 'redis':
            # Redis setup (good for fast access)
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0
            )

        elif storage_type == 'postgresql':
            # PostgreSQL setup (good for strict schema requirements)
            self.engine = create_engine('postgresql://user:password@localhost:5432/ml_models')
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()

        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

    def save_model(self,
                   model: Any,
                   model_id: str,
                   metadata: Dict[str, Any]) -> bool:
        """
        Save a trained model and its metadata to the database
        """
        try:
            if self.storage_type == 'mongodb':
                return self._save_to_mongodb(model, model_id, metadata)
            elif self.storage_type == 'redis':
                return self._save_to_redis(model, model_id, metadata)
            elif self.storage_type == 'postgresql':
                return self._save_to_postgresql(model, model_id, metadata)
        except Exception as e:
            self.logger.error(f"Error saving model {model_id}: {str(e)}")
            raise

    def _save_to_mongodb(self, model: Any, model_id: str, metadata: Dict[str, Any]) -> bool:
        """Save model to MongoDB using GridFS"""
        try:
            # Serialize the model
            model_bytes = pickle.dumps(model)

            # Add timestamp and version to metadata
            metadata.update({
                'created_at': datetime.now(),
                'storage_type': 'mongodb'
            })

            # Store the model in GridFS
            file_id = self.fs.put(
                model_bytes,
                filename=f"model_{model_id}",
                metadata=metadata
            )

            # Store metadata in separate collection for easier querying
            self.db.model_metadata.insert_one({
                'model_id': model_id,
                'file_id': file_id,
                **metadata
            })

            return True

        except Exception as e:
            self.logger.error(f"MongoDB save error: {str(e)}")
            raise

    def _save_to_redis(self, model: Any, model_id: str, metadata: Dict[str, Any]) -> bool:
        """Save model to Redis"""
        try:
            # Serialize the model
            model_bytes = pickle.dumps(model)

            # Add timestamp to metadata
            metadata['created_at'] = datetime.now().isoformat()

            # Store model and metadata
            self.redis_client.set(f"model:{model_id}:data", model_bytes)
            self.redis_client.set(f"model:{model_id}:metadata", json.dumps(metadata))

            return True

        except Exception as e:
            self.logger.error(f"Redis save error: {str(e)}")
            raise

    def _save_to_postgresql(self, model: Any, model_id: str, metadata: Dict[str, Any]) -> bool:
        """Save model to PostgreSQL"""
        try:
            # Serialize the model
            model_bytes = pickle.dumps(model)

            # Create metadata record
            model_metadata = ModelMetadata(
                id=model_id,
                version=metadata.get('version', '1.0'),
                created_at=datetime.now(),
                accuracy=metadata.get('accuracy', 0.0),
                feature_importance=json.dumps(metadata.get('feature_importance', {})),
                training_size=metadata.get('training_size', 0),
                framework=metadata.get('framework', 'sklearn')
            )

            # Store metadata
            self.session.add(model_metadata)

            # Store model binary
            self.session.execute(
                'INSERT INTO model_binary (id, data) VALUES (:id, :data)',
                {'id': model_id, 'data': model_bytes}
            )

            self.session.commit()
            return True

        except Exception as e:
            self.session.rollback()
            self.logger.error(f"PostgreSQL save error: {str(e)}")
            raise

    def load_model(self, model_id: str) -> Optional[Any]:
        """Load a model from the database"""
        try:
            if self.storage_type == 'mongodb':
                return self._load_from_mongodb(model_id)
            elif self.storage_type == 'redis':
                return self._load_from_redis(model_id)
            elif self.storage_type == 'postgresql':
                return self._load_from_postgresql(model_id)
        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {str(e)}")
            raise

    def _load_from_mongodb(self, model_id: str) -> Optional[Any]:
        """Load model from MongoDB"""
        try:
            # Find the latest version of the model
            metadata = self.db.model_metadata.find_one({'model_id': model_id})
            if not metadata:
                return None

            # Get the model from GridFS
            grid_out = self.fs.get(metadata['file_id'])
            return pickle.loads(grid_out.read())

        except Exception as e:
            self.logger.error(f"MongoDB load error: {str(e)}")
            raise

    def _load_from_redis(self, model_id: str) -> Optional[Any]:
        """Load model from Redis"""
        try:
            # Get model data
            model_data = self.redis_client.get(f"model:{model_id}:data")
            if not model_data:
                return None

            return pickle.loads(model_data)

        except Exception as e:
            self.logger.error(f"Redis load error: {str(e)}")
            raise

    def _load_from_postgresql(self, model_id: str) -> Optional[Any]:
        """Load model from PostgreSQL"""
        try:
            # Get model binary data
            result = self.session.execute(
                'SELECT data FROM model_binary WHERE id = :id',
                {'id': model_id}
            ).first()

            if not result:
                return None

            return pickle.loads(result[0])

        except Exception as e:
            self.logger.error(f"PostgreSQL load error: {str(e)}")
            raise


# Example usage with the DeploymentPredictor
def store_trained_model():
    from deployment_predictor import DeploymentPredictor

    # Train the model
    predictor = DeploymentPredictor()
    # ... training code ...

    # Initialize storage system (using MongoDB as recommended)
    storage = ModelStorageSystem(storage_type='mongodb')

    # Prepare metadata
    metadata = {
        'version': '1.0',
        'accuracy': 0.85,
        'feature_importance': {
            'config_type': 0.3,
            'component': 0.2,
            'is_critical': 0.5
        },
        'training_size': 1000,
        'framework': 'sklearn'
    }

    # Save the model
    model_id = 'deployment_predictor_v1'
    storage.save_model(predictor, model_id, metadata)

    # Later, load the model
    loaded_predictor = storage.load_model(model_id)

    return loaded_predictor


if __name__ == "__main__":
    # Example usage
    predictor = store_trained_model()