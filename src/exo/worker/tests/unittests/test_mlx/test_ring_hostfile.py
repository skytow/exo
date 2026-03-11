import pytest
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import BoundInstance, InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.utils_mlx import _init_ring_with_retry, _shared_ring_hosts


def _pipeline_shard(rank: int, world_size: int) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("mlx-community/test-model"),
            storage_size=Memory.from_gb(1),
            n_layers=10,
            hidden_size=1024,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=rank,
        world_size=world_size,
        start_layer=0 if rank == 0 else 5,
        end_layer=5 if rank == 0 else 10,
        n_layers=10,
    )


def test_shared_ring_hosts_derive_single_rank_ordered_list() -> None:
    anton = NodeId("anton")
    smithers = NodeId("smithers")
    anton_runner = RunnerId("anton-runner")
    smithers_runner = RunnerId("smithers-runner")

    instance = MlxRingInstance(
        instance_id=InstanceId("instance"),
        shard_assignments=ShardAssignments(
            model_id=ModelId("mlx-community/test-model"),
            node_to_runner={
                anton: anton_runner,
                smithers: smithers_runner,
            },
            runner_to_shard={
                anton_runner: _pipeline_shard(0, 2),
                smithers_runner: _pipeline_shard(1, 2),
            },
        ),
        hosts_by_node={
            anton: [
                Host(ip="0.0.0.0", port=53612),
                Host(ip="192.168.1.42", port=53612),
            ],
            smithers: [
                Host(ip="192.168.1.35", port=53612),
                Host(ip="0.0.0.0", port=53612),
            ],
        },
        ephemeral_port=53612,
    )

    anton_bound = BoundInstance(
        instance=instance,
        bound_runner_id=anton_runner,
        bound_node_id=anton,
    )
    smithers_bound = BoundInstance(
        instance=instance,
        bound_runner_id=smithers_runner,
        bound_node_id=smithers,
    )

    expected = [
        Host(ip="192.168.1.35", port=53612),
        Host(ip="192.168.1.42", port=53612),
    ]

    assert _shared_ring_hosts(anton_bound) == expected
    assert _shared_ring_hosts(smithers_bound) == expected


def test_init_ring_with_retry_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_MLX_RING_CONNECT_STAGGER_SECONDS", "0")
    monkeypatch.setenv("EXO_MLX_RING_INIT_RETRIES", "3")
    monkeypatch.setenv("EXO_MLX_RING_INIT_BACKOFF_SECONDS", "0.1")

    sleeps: list[float] = []
    monkeypatch.setattr(
        "exo.worker.engines.mlx.utils_mlx.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    attempts = {"count": 0}

    def _init() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("[ring] Couldn't connect (error: 60)")
        return "ok"

    result = _init_ring_with_retry(rank=1, init_fn=_init)

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleeps == [0.1, 0.2]


def test_init_ring_with_retry_does_not_retry_non_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_MLX_RING_CONNECT_STAGGER_SECONDS", "0")
    monkeypatch.setenv("EXO_MLX_RING_INIT_RETRIES", "3")
    monkeypatch.setenv("EXO_MLX_RING_INIT_BACKOFF_SECONDS", "0.1")
    monkeypatch.setattr("exo.worker.engines.mlx.utils_mlx.time.sleep", lambda _: None)

    attempts = {"count": 0}

    def _init() -> str:
        attempts["count"] += 1
        raise RuntimeError("non-ring failure")

    with pytest.raises(RuntimeError, match="non-ring failure"):
        _init_ring_with_retry(rank=0, init_fn=_init)

    assert attempts["count"] == 1
