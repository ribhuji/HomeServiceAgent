from pathlib import Path

from home_service_agent.domain import ServiceRequest, WorkflowPhase, WorkflowState
from home_service_agent.mocks.provider_api import MockProviderAPI
from home_service_agent.storage import initialize_storage
from home_service_agent.workflow import build_graph


def test_initialize_storage_creates_local_database_files(tmp_path: Path) -> None:
    paths = initialize_storage(tmp_path / "data")

    assert paths.agent_db.exists()
    assert paths.checkpoints_db.exists()


def test_sqlite_checkpointer_survives_graph_reconstruction(tmp_path: Path) -> None:
    paths = initialize_storage(tmp_path / "data")
    config = {"configurable": {"thread_id": "persistent-thread"}}
    graph = build_graph(MockProviderAPI(seed=1), checkpoint_path=str(paths.checkpoints_db))
    graph.invoke(
        {
            "workflow": WorkflowState(
                run_id="persistent-thread",
                user_id="user-1",
                request=ServiceRequest(
                    service="plumber",
                    area="Indiranagar",
                    time_window="this week",
                ),
            )
        },
        config,
    )

    restored_graph = build_graph(
        MockProviderAPI(seed=1),
        checkpoint_path=str(paths.checkpoints_db),
    )
    restored = restored_graph.get_state(config)

    workflow = restored.values["workflow"]
    phase = workflow.phase if hasattr(workflow, "phase") else workflow["phase"]
    assert phase is WorkflowPhase.OPTIONS_READY or phase == WorkflowPhase.OPTIONS_READY.value

