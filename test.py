import pytest
# Import main modules for testing
from responder import generate_response
from query import query_helpdesk
from retriever import sanitize_meta

def test_generate_response_format():
    """Test that generate_response returns a properly formatted string."""
    user_input = "How do I reset my password?"
    context = [
        "To reset your password, visit https://reset.techcorp.com and follow the steps.",
        "If you cannot access the reset portal, contact support@techcorp.com."
    ]
    response = generate_response(user_input, context)
    assert isinstance(response, str)
    assert "Category:" in response
    assert "Response:" in response
    assert "Escalation Required:" in response

def test_query_helpdesk_returns_results():
    """Test that query_helpdesk returns a list of (chunk, metadata) tuples."""
    user_input = "I need help with Outlook installation"
    results = query_helpdesk(user_input, top_k=2)
    assert isinstance(results, list)
    assert len(results) > 0
    assert isinstance(results[0], tuple)
    assert isinstance(results[0][0], str)       # document chunk
    assert isinstance(results[0][1], dict)      # metadata

def test_sanitize_meta_removes_invalid_fields():
    """Test that sanitize_meta removes non-primitive fields from metadata."""
    meta = {
        "title": "Test Guide",
        "category": "software",
        "tags": ["test"],
        "updated": "2025-07-22",
        "bad": ["this", "is", "a", "list"]   # ❌ invalid
    }
    clean = sanitize_meta(meta)
    assert "bad" not in clean
    assert all(isinstance(v, (str, int, float, bool, type(None))) for v in clean.values())

# --- Additional unit test stubs for other modules ---
def test_api_server_import():
    """Test that Api_server.py imports without error."""
    try:
        import Api_server
    except Exception as e:
        pytest.fail(f"Api_server import failed: {e}")

# def test_main_import():
#     """Test that main.py imports without error (if present)."""
#     try:
#         import main
#     except Exception as e:
#         pytest.fail(f"main import failed: {e}")

def test_query_module():
    """Test that query.py has query_helpdesk function."""
    import query
    assert hasattr(query, "query_helpdesk")

def test_responder_module():
    """Test that responder.py has generate_response function."""
    import responder
    assert hasattr(responder, "generate_response")

def test_retriever_module():
    """Test that retriever.py has sanitize_meta function."""
    import retriever
    assert hasattr(retriever, "sanitize_meta")