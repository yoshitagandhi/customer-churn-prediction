"""Service layer bridging the Streamlit UI to the core `src` package.

Every function here is plain Python with no Streamlit imports (except
where noted), making the service layer independently testable and
keeping all machine-learning logic in `src/` where it belongs. Pages
call these services; services call `src/`.
"""
