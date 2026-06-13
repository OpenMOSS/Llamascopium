from typing import Any, Optional

from fastapi import APIRouter, Response
from pydantic import BaseModel

from server.config import client, sae_series
from server.logic.samples import cached_extract_samples
from server.logic.loaders import get_sae
from server.utils.common import natural_sort_key

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/cache/clear-samples")
def admin_clear_sample_cache():
    """Clear cached feature activation samples."""
    cached_extract_samples.cache_clear()
    return {"message": "Sample cache cleared"}


@router.get("/saes")
def admin_list_saes(limit: int = 100, skip: int = 0, search: Optional[str] = None):
    """List all SAEs (dictionaries) in the current series with full details."""
    query: dict[str, Any] = {"series": sae_series}
    if search:
        query["name"] = {"$regex": search, "$options": "i"}

    total_count = client.sae_collection.count_documents(query)
    saes = client.sae_collection.find(query).sort("name", 1).skip(skip).limit(limit)

    results = []
    for sae in saes:
        sae_name = sae["name"]
        # Get analysis count
        analysis_count = client.analysis_collection.count_documents({"sae_name": sae_name, "sae_series": sae_series})
        # Get feature count
        feature_count = client.feature_collection.count_documents({"sae_name": sae_name, "sae_series": sae_series})

        results.append(
            {
                "name": sae_name,
                "series": sae["series"],
                "path": sae.get("path"),
                "model_name": sae.get("model_name"),
                "analysis_count": analysis_count,
                "feature_count": feature_count,
                "cfg": sae.get("cfg"),
            }
        )
    return {"saes": results, "total_count": total_count}


class UpdateSaeRequest(BaseModel):
    new_name: Optional[str] = None
    path: Optional[str] = None
    model_name: Optional[str] = None


@router.put("/saes/{name}")
def admin_update_sae(name: str, request: UpdateSaeRequest):
    """Update an SAE's metadata, including renaming."""
    # Check if SAE exists
    sae = client.sae_collection.find_one({"name": name, "series": sae_series})
    if sae is None:
        return Response(content=f"SAE {name} not found", status_code=404)

    update_data: dict[str, Any] = {}
    if request.path is not None:
        update_data["path"] = request.path
    if request.model_name is not None:
        update_data["model_name"] = request.model_name

    # Handle renaming
    if request.new_name is not None and request.new_name != name:
        # Check if new name already exists
        existing = client.sae_collection.find_one({"name": request.new_name, "series": sae_series})
        if existing:
            return Response(content=f"SAE with name '{request.new_name}' already exists", status_code=409)

        update_data["name"] = request.new_name
        # Clear cache
        get_sae.cache_clear()

    if not update_data:
        return {"message": "No updates provided"}

    client.update_sae(name, sae_series, update_data)

    new_name = request.new_name if request.new_name else name
    return {"message": f"SAE '{new_name}' updated successfully"}


@router.delete("/saes/{name}")
def admin_delete_sae(name: str):
    """Delete an SAE and all its associated data (features, analyses)."""
    # Check if SAE exists
    sae = client.sae_collection.find_one({"name": name, "series": sae_series})
    if sae is None:
        return Response(content=f"SAE {name} not found", status_code=404)

    # Delete all associated data
    client.remove_sae_analysis(sae_name=name, sae_series=sae_series)

    # Clear cache
    get_sae.cache_clear()

    return {"message": f"SAE '{name}' and all associated data deleted successfully"}


@router.get("/sae-sets")
def admin_list_sae_sets():
    """List all SAE sets with full details."""
    sae_sets = client.sae_set_collection.find({"sae_series": sae_series})
    results = []
    for sae_set in sae_sets:
        results.append(
            {
                "name": sae_set["name"],
                "sae_series": sae_set["sae_series"],
                "sae_names": sae_set["sae_names"],
            }
        )
    return sorted(results, key=lambda x: natural_sort_key(x["name"]))


class UpdateSaeSetRequest(BaseModel):
    new_name: Optional[str] = None
    sae_names: Optional[list[str]] = None


@router.put("/sae-sets/{name}")
def admin_update_sae_set(name: str, request: UpdateSaeSetRequest):
    """Update an SAE set, including renaming."""
    # Check if SAE set exists
    sae_set = client.sae_set_collection.find_one({"name": name})
    if sae_set is None:
        return Response(content=f"SAE set {name} not found", status_code=404)

    update_data: dict[str, Any] = {}
    if request.sae_names is not None:
        update_data["sae_names"] = request.sae_names

    # Handle renaming
    if request.new_name is not None and request.new_name != name:
        # Check if new name already exists
        existing = client.sae_set_collection.find_one({"name": request.new_name})
        if existing:
            return Response(content=f"SAE set with name '{request.new_name}' already exists", status_code=409)

        update_data["name"] = request.new_name

    if not update_data:
        return {"message": "No updates provided"}

    client.update_sae_set(name, update_data)

    new_name = request.new_name if request.new_name else name
    return {"message": f"SAE set '{new_name}' updated successfully"}


@router.delete("/sae-sets/{name}")
def admin_delete_sae_set(name: str):
    """Delete an SAE set."""
    result = client.sae_set_collection.delete_one({"name": name})

    if result.deleted_count == 0:
        return Response(content=f"SAE set {name} not found", status_code=404)

    return {"message": f"SAE set '{name}' deleted successfully"}


@router.get("/circuits")
def admin_list_circuits(limit: int = 100, skip: int = 0):
    """List all circuits with full details for admin."""
    circuits = client.list_circuits(sae_series=sae_series, limit=limit, skip=skip)

    total_count = client.circuit_collection.count_documents({"sae_series": sae_series})

    return {"circuits": circuits, "total_count": total_count}


class UpdateCircuitRequest(BaseModel):
    name: Optional[str] = None
    group: Optional[str] = None


@router.put("/circuits/{circuit_id}")
def admin_update_circuit(circuit_id: str, request: UpdateCircuitRequest):
    """Update a circuit's metadata."""
    from bson import ObjectId

    update_data = {}
    if request.name is not None:
        update_data["name"] = request.name
    if request.group is not None:
        update_data["group"] = request.group

    if not update_data:
        return {"message": "No updates provided"}

    try:
        result = client.circuit_collection.update_one({"_id": ObjectId(circuit_id)}, {"$set": update_data})
    except Exception:
        return Response(content=f"Invalid circuit ID: {circuit_id}", status_code=400)

    if result.matched_count == 0:
        return Response(content=f"Circuit {circuit_id} not found", status_code=404)

    return {"message": "Circuit updated successfully"}


class BulkGroupRequest(BaseModel):
    circuit_ids: list[str]
    group: Optional[str] = None


@router.post("/circuits/bulk-group")
def admin_bulk_group_circuits(request: BulkGroupRequest):
    """Update the group for multiple circuits."""
    count = client.update_circuits_group(request.circuit_ids, request.group)
    return {"message": f"Updated {count} circuits"}


@router.get("/stats")
def admin_get_stats():
    """Get overall statistics for the admin dashboard."""
    sae_count = client.sae_collection.count_documents({"series": sae_series})
    sae_set_count = client.sae_set_collection.count_documents({"sae_series": sae_series})
    circuit_count = client.circuit_collection.count_documents({"sae_series": sae_series})
    bookmark_count = client.bookmark_collection.count_documents({"sae_series": sae_series})

    return {
        "sae_count": sae_count,
        "sae_set_count": sae_set_count,
        "circuit_count": circuit_count,
        "bookmark_count": bookmark_count,
        "sae_series": sae_series,
    }
