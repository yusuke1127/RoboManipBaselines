def remove_suffix(s, suffix):
    """Support Python 3.8 and below."""
    if s.endswith(suffix):
        return s[: -len(suffix)]
    return s
