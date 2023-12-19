from unittest.mock import MagicMock, create_autospec, patch

from langchain.llms.fake import FakeListLLM
from slackapp.bolt_app import get_response


@patch("sherpa_ai.agents.qa_agent.QAAgent.run")
@patch("sherpa_ai.task_agent.TaskAgent.run")
def test_normal_mode(mock_task_agent_run, mock_qa_agent_run):
    mock_task_agent_run.return_value = "Task agent response"
    mock_qa_agent_run.return_value = "QA agent response"

    verbose_logger = MagicMock()
    llm = create_autospec(FakeListLLM)

    _ = get_response(
        question="Test message",
        previous_messages=[],
        verbose_logger=verbose_logger,
        bot_info={"user_id": "Sherpa"},
        llm=llm,
    )

    mock_qa_agent_run.assert_called()
    mock_task_agent_run.assert_not_called()


@patch("sherpa_ai.agents.qa_agent.QAAgent.run")
@patch("sherpa_ai.task_agent.TaskAgent.run")
def test_obsolete_mode(mock_task_agent_run, mock_qa_agent_run):
    mock_task_agent_run.return_value = "Task agent response"
    mock_qa_agent_run.return_value = "QA agent response"

    verbose_logger = MagicMock()
    llm = create_autospec(FakeListLLM)

    _ = get_response(
        question="Messages? --obsolete",
        previous_messages=[],
        verbose_logger=verbose_logger,
        bot_info={"user_id": "Sherpa"},
        llm=llm,
    )

    mock_qa_agent_run.assert_not_called()
    mock_task_agent_run.assert_called()
