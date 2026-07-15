import mongomock
import pytest

from llamascopium import MongoDBConfig, SAEConfig
from llamascopium.database import MongoClient, SAERecord


@pytest.fixture
def mongo_client(mocker) -> MongoClient:
    """
    Creates a MongoClient instance with an in-memory MongoDB.

    Returns:
        MongoClient: A configured MongoClient instance for testing
    """
    mock_gridfs = mocker.patch("gridfs.GridFS")
    mock_gridfs.return_value.exists.return_value = False
    with mongomock.patch(servers=(("fake", 27017),)):
        client = MongoClient(MongoDBConfig(mongo_uri="mongodb://fake", mongo_db="test_db"))
        yield client
        # Clear all collections after each test
        client.db.drop_collection("sae")
        client.db.drop_collection("analysis")
        client.db.drop_collection("feature")


def test_create_and_get_sae(mongo_client: MongoClient) -> None:
    """Test creating and retrieving an SAE record."""
    # Arrange
    name = "test_sae"
    series = "test_series"
    path = "test_path"
    cfg = SAEConfig(
        hook_point_in="test_hook_point_in",
        hook_point_out="test_hook_point_out",
        d_sae=10,
        d_model=10,
        expansion_factor=1,
    )

    # Act
    mongo_client.create_sae(name, series, path, cfg)
    result = mongo_client.get_sae(name, series)

    # Assert
    assert isinstance(result, SAERecord)
    assert result.name == name
    assert result.series == series
    assert result.path == path
    assert result.cfg.d_sae == cfg.d_sae


def test_upsert_sae_metadata_preserves_existing_features(mongo_client: MongoClient) -> None:
    cfg = SAEConfig(
        hook_point_in="test_hook_point_in",
        hook_point_out="test_hook_point_out",
        d_model=2,
        expansion_factor=2,
    )
    feature = {
        "sae_name": "test_sae",
        "sae_series": "test_series",
        "index": 0,
        "analyses": [{"name": "existing-analysis"}],
    }
    mongo_client.feature_collection.insert_one(feature)

    mongo_client.upsert_sae_metadata(
        "test_sae",
        "test_series",
        "first_path",
        cfg,
        model_name="test-model",
    )
    mongo_client.upsert_sae_metadata(
        "test_sae",
        "test_series",
        "updated_path",
        cfg,
        model_name="test-model",
    )

    assert mongo_client.feature_collection.count_documents({}) == 1
    stored_feature = mongo_client.feature_collection.find_one({"sae_name": "test_sae"})
    assert stored_feature is not None
    assert stored_feature["analyses"] == [{"name": "existing-analysis"}]
    stored_sae = mongo_client.sae_collection.find_one({"name": "test_sae", "series": "test_series"})
    assert stored_sae is not None
    assert stored_sae["path"] == "updated_path"
    assert stored_sae["model_name"] == "test-model"


def test_list_saes(mongo_client: MongoClient) -> None:
    """Test listing SAE records."""
    # Arrange
    cfg = SAEConfig(
        hook_point_in="test_hook_point_in",
        hook_point_out="test_hook_point_out",
        d_sae=10,
        d_model=10,
        expansion_factor=1,
    )
    print(mongo_client.sae_collection.find_one())
    mongo_client.create_sae("sae1", "series1", "test_path", cfg)
    mongo_client.create_sae("sae2", "series1", "test_path", cfg)
    mongo_client.create_sae("sae3", "series2", "test_path", cfg)

    # Act & Assert
    assert set(mongo_client.list_saes()) == {"sae1", "sae2", "sae3"}
    assert set(mongo_client.list_saes("series1")) == {"sae1", "sae2"}
    assert set(mongo_client.list_saes("series2")) == {"sae3"}


def test_remove_sae(mongo_client: MongoClient) -> None:
    """Test removing an SAE record."""
    # Arrange
    name = "test_sae"
    series = "test_series"
    path = "test_path"
    cfg = SAEConfig(
        hook_point_in="test_hook_point_in",
        hook_point_out="test_hook_point_out",
        d_sae=10,
        d_model=10,
        expansion_factor=1,
    )
    mongo_client.create_sae(name, series, path, cfg)

    # Act
    mongo_client.remove_sae(name, series)

    # Assert
    assert mongo_client.get_sae(name, series) is None
    assert mongo_client.list_saes(series) == []


def test_create_and_get_analysis(mongo_client: MongoClient) -> None:
    """Test creating and retrieving analysis records."""
    # Arrange
    cfg = SAEConfig(
        hook_point_in="test_hook_point_in",
        hook_point_out="test_hook_point_out",
        d_sae=10,
        d_model=10,
        expansion_factor=1,
    )
    mongo_client.create_sae("test_sae", "test_series", "test_path", cfg)

    # Act
    mongo_client.create_analysis("test_analysis", "test_sae", "test_series")
    analyses = mongo_client.list_analyses("test_sae", "test_series")

    # Assert
    assert analyses == ["test_analysis"]


def test_get_nonexistent_sae(mongo_client: MongoClient) -> None:
    """Test retrieving a non-existent SAE record."""
    assert mongo_client.get_sae("nonexistent", "series") is None


def test_get_feature(mongo_client: MongoClient) -> None:
    """Test retrieving feature records."""
    # Arrange
    cfg = SAEConfig(
        hook_point_in="test_hook_point_in", hook_point_out="test_hook_point_out", d_sae=2, d_model=2, expansion_factor=1
    )
    mongo_client.create_sae("test_sae", "test_series", "test_path", cfg)

    # Act
    feature = mongo_client.get_feature("test_sae", "test_series", 0)

    # Assert
    assert feature is not None
    assert feature.sae_name == "test_sae"
    assert feature.sae_series == "test_series"
    assert feature.index == 0
