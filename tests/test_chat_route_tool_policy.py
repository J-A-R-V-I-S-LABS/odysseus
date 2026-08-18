"""Issue #3229 and explicit web-toggle regressions.

Bug: allow_bash and allow_web_search were only read from form_data, so JSON
API callers (Content-Type: application/json) always had bash disabled.

Fix: (1) Read from JSON body as fallback.
     (2) Keep bash on the privilege fallback when unset.
     (3) Require an explicit per-turn web setting before exposing web tools.
"""

import ast
import re
from pathlib import Path

import pytest

from src.action_intents import classify_tool_intent
from src.tool_policy import (
    WEB_TOOL_NAMES,
    is_web_search_explicitly_denied,
    web_search_enabled_for_turn,
)

_CHAT_ROUTES = Path(__file__).resolve().parent.parent / "routes" / "chat_routes.py"


# ── Source-level guards ─────────────────────────────────────────


def test_allow_bash_reads_from_body_as_fallback():
    """chat_stream must read allow_bash from the JSON body, not just form_data."""
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find the chat_stream function
    chat_stream_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "chat_stream":
            chat_stream_func = node
            break
    assert chat_stream_func is not None, "chat_stream function not found"

    # Look for an assignment to allow_bash that references 'body'
    found_body_fallback = False
    for node in ast.walk(chat_stream_func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "allow_bash":
                    # Check if 'body' appears in the value
                    src_segment = ast.get_source_segment(source, node)
                    if src_segment and "body" in src_segment:
                        found_body_fallback = True
    assert found_body_fallback, (
        "allow_bash assignment in chat_stream must fall back to JSON body"
    )


def test_allow_web_search_reads_from_body_as_fallback():
    """chat_stream must read allow_web_search from the JSON body, not just form_data."""
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    tree = ast.parse(source)

    chat_stream_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "chat_stream":
            chat_stream_func = node
            break
    assert chat_stream_func is not None

    found_body_fallback = False
    for node in ast.walk(chat_stream_func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "allow_web_search":
                    src_segment = ast.get_source_segment(source, node)
                    if src_segment and "body" in src_segment:
                        found_body_fallback = True
    assert found_body_fallback, (
        "allow_web_search assignment in chat_stream must fall back to JSON body"
    )


def test_browser_form_followups_include_approval_and_send_phrases():
    """Short approval replies after a form/browser turn must keep browser tools available."""
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    assert "approved" in source
    assert "proceed" in source
    assert "send(?:\\s+it)?" in source
    assert "submit(?:\\s+it)?" in source


def test_agent_loop_expands_browser_mcp_tools_from_connected_server():
    """Browser intent must not depend on stale hardcoded Playwright tool names."""
    source = (Path(__file__).resolve().parent.parent / "src" / "agent_loop.py").read_text(encoding="utf-8")
    assert "def _expand_browser_mcp_tools" in source
    assert "server_id\") == \"builtin_browser\"" in source
    assert "_relevant_tools = _expand_browser_mcp_tools(_relevant_tools, mcp_mgr)" in source


def test_disabled_tools_respects_missing_vs_explicit_toggles():
    """Bash still defers to privileges, but web is an explicit per-turn opt-in.
    """
    source = _CHAT_ROUTES.read_text(encoding="utf-8")

    # The fix changes:
    #   if str(allow_bash).lower() != "true":
    # to:
    #   if allow_bash is not None and str(allow_bash).lower() != "true":
    assert "allow_bash is not None" in source, (
        "disabled_tools check must guard against allow_bash being None"
    )
    assert "web_search_enabled_for_turn(allow_web_search, use_web)" in source, (
        "web tools must be gated through the explicit per-turn web setting"
    )
    assert "disabled_tools.update(WEB_TOOL_NAMES)" in source, (
        "disabled_tools must add web_search/web_fetch when web is not explicitly enabled"
    )
    assert "_forced_tools = set(WEB_TOOL_NAMES)" in source, (
        "web tools should only be forced visible from the explicit web setting"
    )


def test_workspace_auto_escalation_keeps_shell_tools():
    """Workspace/shell auto-routing must not use the light typed-tool clamp."""
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    assert '_workspace_agent_intent = _tool_intent.category in {"shell", "workspace"}' in source
    assert "allow_bash = \"true\"" in source
    assert "if auto_escalated and not _workspace_agent_intent:" in source


# ── Functional tests of the disabled-tools logic ───────────────


def _build_disabled_tools(
    allow_bash=None,
    allow_web_search=None,
    use_web=None,
    can_use_bash=True,
    can_use_browser=True,
    explicit_web_intent=False,
    local_task_intent=False,
    global_disabled=None,
):
    """Replicate the disabled-tools logic from chat_stream for unit testing.

    Returns the set of tool names that would be disabled.
    """
    disabled_tools = set()

    # Issue #3229 fix: only disable bash when explicitly set to a falsy value.
    if allow_bash is not None and str(allow_bash).lower() != "true":
        disabled_tools.add("bash")
    search_enabled = web_search_enabled_for_turn(allow_web_search, use_web)
    if is_web_search_explicitly_denied(allow_web_search) or not search_enabled:
        disabled_tools.update(WEB_TOOL_NAMES)
    if explicit_web_intent and not local_task_intent:
        disabled_tools.update({
            "bash", "python",
            "search_chats", "manage_skills", "manage_memory",
            "read_file", "write_file", "edit_file",
            "create_document", "edit_document", "update_document",
            "send_email", "reply_to_email",
            "manage_notes", "manage_calendar", "manage_tasks",
            "api_call", "builtin_browser",
        })
        if search_enabled:
            disabled_tools.difference_update(WEB_TOOL_NAMES)
        else:
            disabled_tools.update(WEB_TOOL_NAMES)
    elif search_enabled:
        disabled_tools.difference_update(WEB_TOOL_NAMES)

    # Enforce per-user privileges
    if not can_use_bash:
        disabled_tools.update({"bash", "python", "read_file", "write_file"})
    if not can_use_browser:
        disabled_tools.add("builtin_browser")
    if global_disabled and isinstance(global_disabled, list):
        disabled_tools.update(global_disabled)

    return disabled_tools


def test_json_body_allow_bash_true_enables_bash():
    """API caller sending {"allow_bash": true} gets bash enabled."""
    disabled = _build_disabled_tools(allow_bash="true")
    assert "bash" not in disabled


def test_json_body_allow_bash_false_disables_bash():
    """API caller sending {"allow_bash": false} gets bash disabled."""
    disabled = _build_disabled_tools(allow_bash="false")
    assert "bash" in disabled


def test_json_body_allow_web_search_true_enables_web():
    """API caller sending {"allow_web_search": true} gets web tools enabled."""
    disabled = _build_disabled_tools(allow_web_search="true")
    assert "web_search" not in disabled
    assert "web_fetch" not in disabled


def test_json_body_allow_web_search_false_disables_web():
    """API caller sending {"allow_web_search": false} gets web tools disabled."""
    disabled = _build_disabled_tools(allow_web_search="false")
    assert "web_search" in disabled
    assert "web_fetch" in disabled


def test_chat_mode_use_web_true_enables_web():
    """Chat pre-search sends use_web=true as the explicit web setting."""
    disabled = _build_disabled_tools(use_web="true")
    assert "web_search" not in disabled
    assert "web_fetch" not in disabled


def test_allow_web_search_false_wins_over_use_web_true():
    """The agent web toggle hard-denies web even if another path says use_web=true."""
    disabled = _build_disabled_tools(allow_web_search="false", use_web="true")
    assert "web_search" in disabled
    assert "web_fetch" in disabled


@pytest.mark.parametrize(
    "message",
    [
        "please use web search for current CVEs",
        "search the web for current CVEs",
        "can you look up the latest docs",
    ],
)
def test_explicit_false_disables_web_despite_prompt_web_intent(message):
    """Explicit allow_web_search=false is a hard deny even when the prompt
    asks for web search."""
    intent = classify_tool_intent(message)
    assert intent is not None
    assert intent.category == "web"

    disabled = _build_disabled_tools(
        allow_web_search="false",
        explicit_web_intent=True,
    )
    assert "web_search" in disabled
    assert "web_fetch" in disabled


def test_prompt_web_intent_does_not_enable_web_without_setting():
    """Prompt-derived web intent alone must not expose web tools."""
    intent = classify_tool_intent("look up the latest docs")
    assert intent is not None
    assert intent.category == "web"

    disabled = _build_disabled_tools(
        allow_web_search=None,
        use_web=None,
        explicit_web_intent=True,
    )
    assert "web_search" in disabled
    assert "web_fetch" in disabled


def test_admin_user_gets_bash_enabled_by_default():
    """When allow_bash is not set and user has can_use_bash privilege,
    bash must NOT be disabled.
    """
    disabled = _build_disabled_tools(allow_bash=None, can_use_bash=True)
    assert "bash" not in disabled


def test_web_search_disabled_by_default_without_explicit_turn_setting():
    """Missing web settings must not expose web tools by default."""
    disabled = _build_disabled_tools(allow_web_search=None)
    assert "web_search" in disabled
    assert "web_fetch" in disabled


def test_non_privileged_user_without_explicit_flag_still_disabled():
    """A user without can_use_bash privilege who doesn't send allow_bash
    should still have bash disabled via the privilege check.
    """
    disabled = _build_disabled_tools(allow_bash=None, can_use_bash=False)
    assert "bash" in disabled


def test_non_privileged_user_explicit_true_overridden_by_privilege():
    """Even if allow_bash=true is sent, a user without can_use_bash
    privilege still gets bash disabled by the privilege gate.
    """
    disabled = _build_disabled_tools(allow_bash="true", can_use_bash=False)
    assert "bash" in disabled


def test_global_disabled_web_wins_over_explicit_web_enable():
    """Admin-level disabled tools are still a hard deny."""
    disabled = _build_disabled_tools(
        allow_web_search="true",
        global_disabled=["web_search", "web_fetch"],
    )
    assert "web_search" in disabled
    assert "web_fetch" in disabled


def test_web_search_denial_regex_exists_and_guards_explicit_web_intent():
    """Source guard: the crude keyword scan that sets _explicit_web_intent
    must be gated by a denial-aware regex, or a turn that tells the agent
    NOT to use web search (which still contains the bare words "web"/
    "search") gets misread as a web-lookup request and clamps bash/python/
    read_file/write_file off — even when those tools were explicitly
    enabled and the message asked for a workspace/file task.
    """
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    assert "_WEB_SEARCH_DENIAL_RE" in source
    assert "not _WEB_SEARCH_DENIAL_RE.search(_msg_l)" in source


def test_web_search_denial_regex_matches_common_negations():
    """The denial regex itself must catch the common ways a user tells the
    agent not to search the web, without over-matching a genuine request.
    """
    from routes.chat_routes import _WEB_SEARCH_DENIAL_RE

    denials = [
        "Inspect the workspace and read requirements.txt. Do not use web search.",
        "Please read requirements.txt. No web search allowed, no writes.",
        "Don't search the web for this, just check the local files.",
        "without doing a web search, summarize the repo",
    ]
    for message in denials:
        assert _WEB_SEARCH_DENIAL_RE.search(message.lower()), message

    allowed = [
        "search the web for the current weather in Boston",
        "please look it up online",
        "what's the latest exchange rate for USD to EUR",
    ]
    for message in allowed:
        assert not _WEB_SEARCH_DENIAL_RE.search(message.lower()), message


def test_workspace_prompt_denying_web_search_keeps_shell_and_file_tools():
    """End-to-end regression for the reported bug: a workspace/file-inspection
    prompt that explicitly says "no web search" must not clamp bash/python/
    read_file/write_file down to web-only, since the literal word "web"
    appears only inside the denial.
    """
    from routes.chat_routes import _WEB_SEARCH_DENIAL_RE

    message = (
        "Inspect the active workspace and read requirements.txt. "
        "Do not use web search. Do not write any files."
    )
    msg_l = message.lower()
    explicit_web_intent = bool(re.search(
        r"\b(search|look\s*up|lookup|google|browse|web|online|latest|current|today|news|weather|forecast|rate|exchange\s+rate)\b",
        msg_l,
    )) and not _WEB_SEARCH_DENIAL_RE.search(msg_l)
    assert explicit_web_intent is False

    disabled = _build_disabled_tools(
        allow_web_search="true",
        can_use_bash=True,
        explicit_web_intent=explicit_web_intent,
    )
    assert "bash" not in disabled
    assert "python" not in disabled
    assert "read_file" not in disabled


def test_local_task_intent_guard_exists_on_web_only_clamp():
    """Source guard: the web-only clamp (ODY-TOOL-ROUTING-02) must not fire
    when the same turn also shows shell/workspace intent, or a legitimate
    mixed local+web task (read a file, verify something on the web, write a
    report) loses its local/code tools even though they're enabled.
    """
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    assert "_local_task_intent" in source
    assert 'if _explicit_web_intent and not _local_task_intent:' in source


def test_web_only_positive_intent_still_clamps_to_web_tools():
    """A genuine web-only lookup (no local/workspace signal) keeps the
    existing, intentional web-only restriction.
    """
    from src.action_intents import classify_tool_intent

    message = "Search the web for the latest Python release."
    intent = classify_tool_intent(message)
    local_task_intent = bool(intent and intent.category in {"shell", "workspace"})
    assert local_task_intent is False

    disabled = _build_disabled_tools(
        allow_web_search="true",
        can_use_bash=True,
        explicit_web_intent=True,
        local_task_intent=local_task_intent,
    )
    assert "bash" in disabled
    assert "python" in disabled
    assert "read_file" in disabled
    assert "web_search" not in disabled


def test_mixed_local_and_web_intent_keeps_both_tool_sets():
    """ODY-TOOL-ROUTING-02 regression: a mixed task (read local file, verify
    something via an authoritative web source, write a report) must keep
    bash/python/read_file/write_file AND web_search — not collapse to
    web_search only.
    """
    from src.action_intents import classify_tool_intent

    message = (
        "Read requirements.txt from the workspace, calculate the required "
        "value, verify the Python version using official web documentation, "
        "and write final_report.txt."
    )
    intent = classify_tool_intent(message)
    assert intent.category in {"shell", "workspace"}, intent
    local_task_intent = True
    explicit_web_intent = bool(re.search(
        r"\b(search|look\s*up|lookup|google|browse|web|online|latest|current|today|news|weather|forecast|rate|exchange\s+rate)\b",
        message.lower(),
    ))
    assert explicit_web_intent is True  # message mentions "web documentation"

    disabled = _build_disabled_tools(
        allow_web_search="true",
        can_use_bash=True,
        explicit_web_intent=explicit_web_intent,
        local_task_intent=local_task_intent,
    )
    assert "bash" not in disabled
    assert "python" not in disabled
    assert "read_file" not in disabled
    assert "write_file" not in disabled
    assert "web_search" not in disabled


def test_mixed_intent_does_not_override_explicitly_disabled_tool():
    """A tool the admin/user explicitly disabled must stay disabled even for
    a mixed local+web task — mixed-intent routing must not widen access.
    """
    disabled = _build_disabled_tools(
        allow_web_search="true",
        can_use_bash=True,
        explicit_web_intent=True,
        local_task_intent=True,
        global_disabled=["write_file"],
    )
    assert "write_file" in disabled
    assert "bash" not in disabled
    assert "read_file" not in disabled


def test_form_data_none_body_true_works():
    """Simulates: form_data has no allow_bash, body has allow_bash=true.
    After the fallback (`form_data.get(...) or body.get(...)`), allow_bash
    should be "true".
    """
    # Simulate the fallback logic
    form_data_val = None  # not in form_data
    body_val = "true"     # from JSON body
    allow_bash = form_data_val or body_val
    assert str(allow_bash).lower() == "true"

    disabled = _build_disabled_tools(allow_bash=allow_bash)
    assert "bash" not in disabled


def test_explicit_false_disables_even_for_admin():
    """An admin who explicitly sends allow_bash=false should have bash disabled."""
    disabled = _build_disabled_tools(
        allow_bash="false", can_use_bash=True,
    )
    assert "bash" in disabled


# ── Frontend source-level guards ──────────────────────────────

_CHAT_JS = Path(__file__).resolve().parent.parent / "static" / "js" / "chat.js"


def test_frontend_always_sends_explicit_allow_bash():
    """chat.js must always send allow_bash (both true and false), not only on toggle ON."""
    source = _CHAT_JS.read_text(encoding="utf-8")
    # Must not only append 'true' — must also handle the false case
    assert "allow_bash', el('bash-toggle').checked ? 'true' : 'false'" in source or \
           "allow_bash', 'false'" in source, (
        "Frontend must send explicit allow_bash=false when toggle is off"
    )


def test_frontend_sends_explicit_allow_web_search_false_in_agent_mode():
    """chat.js must send allow_web_search=false when web toggle is off in agent mode."""
    source = _CHAT_JS.read_text(encoding="utf-8")
    assert "fd.append('allow_web_search', el('web-toggle').checked ? 'true' : 'false')" in source, (
        "Frontend must send explicit allow_web_search=false in agent mode when toggle is off"
    )
